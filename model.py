import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args= None):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet # Batch_size * maxlen * embed_dim

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # Batch_size * maxlen * embed_dim
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) # Batch_size * maxlen * embed_dim


        pos_logits = (log_feats * pos_embs) # Batch_size * maxlen * embed_dim
        pos_logits = pos_logits.sum(dim=-1) # Batch_size * maxlen
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # Batch_size * maxlen
        # import pdb; pdb.set_trace()

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet  # (batch, length, 256)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste(batch, 256)

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (101, 256) '101 = 1pos+100neg'

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        

        return logits # preds # (batch, 101)  '101 = 1pos+100neg'


class GRU4Rec(nn.Module):
    """
    d_model - the number of expected features in the input
    nhead - the number of heads in the multiheadattention models
    dim_feedforward - the hidden dimension size of the feedforward network model
    """
    def __init__(self, user_num, item_num, args, batch_first=True, max_length=50, pad_token=0):
        
        super(GRU4Rec, self).__init__()
        self.hidden_dim = args.hidden_units
        self.batch_first = batch_first
        self.item_num = item_num
        self.max_length = max_length
        self.pad_token = pad_token
        self.dev = args.device
        self.item_emb = nn.Embedding(item_num+1, self.hidden_dim, padding_idx=pad_token)

        self.encoder_layer = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=self.batch_first)

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        log_feats = self.encoder_layer(seqs)  
        return log_feats[0]

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs,  pack=True):
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet # Batch_size * maxlen * embed_dim

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # Batch_size * maxlen * embed_dim
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) # Batch_size * maxlen * embed_dim


        pos_logits = (log_feats * pos_embs) # Batch_size * maxlen * embed_dim
        pos_logits = pos_logits.sum(dim=-1) # Batch_size * maxlen
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # Batch_size * maxlen
        return pos_logits, neg_logits # pos_pred, neg_pred
    
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet  # (batch, length, 256)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste(batch, 256)

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (101, 256) '101 = 1pos+100neg'

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (batch, 101)  '101 = 1pos+100neg'



activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}

class Caser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang , WSDM '18

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, args):
        super(Caser, self).__init__()
        ########### init args
        L = args.maxlen
        dims = args.hidden_units #  # raw default:50
        self.n_h = 4
        self.n_v = 2
        self.drop_ratio = args.dropout_rate
        self.ac_conv = activation_getter['relu']
        self.ac_fc = activation_getter['relu']
        self.dev = args.device

        # user and item embeddings
        self.item_emb = nn.Embedding(num_items + 1, dims)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        self.fc1_dim_v = self.n_v * dims
        self.fc_v = nn.Linear(self.fc1_dim_v, dims)
        self.fc_h = nn.Linear(self.n_h, dims)


    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        item_embs = seqs.unsqueeze(1) 

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            # print(out_v.shape)
            out_v = out_v.view(-1, self.fc1_dim_v).unsqueeze(1)  # b, 1024

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                # import pdb; pdb.set_trace()
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out.unsqueeze(1))
            out_h = torch.cat(out_hs, 1)  # b, length, 16

        temp_v = self.fc_v(out_v)
        temp_h = self.fc_h(out_h)
        log_feats = temp_h + temp_v
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs,  pack=True):
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet # Batch_size * maxlen * embed_dim


        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev)) # Batch_size * maxlen * embed_dim
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev)) # Batch_size * maxlen * embed_dim
        # import pdb; pdb.set_trace()

        pos_logits = (log_feats * pos_embs) # Batch_size * maxlen * embed_dim
        pos_logits = pos_logits.sum(dim=-1) # Batch_size * maxlen
        neg_logits = (log_feats * neg_embs).sum(dim=-1) # Batch_size * maxlen
        return pos_logits, neg_logits # pos_pred, neg_pred
    
    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet  # (batch, length, 256)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste(batch, 256)

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (101, 256) '101 = 1pos+100neg'

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits # preds # (batch, 101)  '101 = 1pos+100neg'



      
