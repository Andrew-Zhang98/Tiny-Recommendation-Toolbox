# Tiny Recommendation Toolbox in PyTorch

A PyTorch implementation of some classic deep learning models for Recommendation.

## Requirements
- Python 3.7
- PyTorch 1.7.0
- numpy 1.20.0

## Models
* Self-attentive sequential recommendation(SASRec)
* Convolutional Sequence Embedding Recommendation Model (Caser) 
* Session-based recommendations with recurrent neural networks(GRU4Rec)

## Training
```
python main.py --model model_name --dataset path_to_your_data --train_dir save_folder 
```

## Citation
If you use this **SASRec** in your paper, please cite the paper:
```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

If you use this **Caser** in your paper, please cite the paper:

```
@inproceedings{tang2018caser,
  title={Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding},
  author={Tang, Jiaxi and Wang, Ke},
  booktitle={ACM International Conference on Web Search and Data Mining},
  year={2018}
}
```
If you use this **GRU4Rec** in your paper, please cite the paper:

```
@article{hidasi2015session,
  title={Session-based recommendations with recurrent neural networks},
  author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros and Baltrunas, Linas and Tikk, Domonkos},
  journal={arXiv preprint arXiv:1511.06939},
  year={2015}
}
```

## Comments

This PyTorch project is tailored for a personal dataset collected by R. Zhao as a gift for the 20th of May, 2021.

Code for other models and public benchmarks will be avaliable soon.
# Acknowledgment

This project is heavily built on [SASRec_pytorch]([https://github.com/pmixer/SASRec.pytorch]). Thanks [Huang Zan](https://github.com/pmixer) for his great work.
