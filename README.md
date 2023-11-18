## FPFNet
The python code implementation of the paper "Feature Pyramid Fusion Network for Hyperspectral Pansharpening"
![image](https://github.com/Jiahuiqu/FPFNet/assets/78287811/23c7dacf-efff-4588-90c6-cdd5c1c9a8cd)


## Requirements
  * Ubuntu 20.04 cuda 11.0
  * Python 3.7 Pytorch 1.10

To install requirements: pip install -r requirements.txt

## Brief description
The train.py include training code and some parameters, detailed in train.py.

The model.py include main model structure.

## dataset
    the default dataset path can be change in dataloader.py
          ─Houston
            ├─train
            │  ├─gtHS
            │  │  ├─1.mat
            │  │  ├─2.mat
            │  │  └─3.mat
            │  ├─hrMS
            │  ├─LRHS
            │  └─LRHS_**
            └─test


## Reference
If you find this code helpful, please kindly cite:

@ARTICLE{10298274,
  author={Dong, Wenqian and Yang, Yihan and Qu, Jiahui and Li, Yunsong and Yang, Yufei and Jia, Xiuping},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Feature Pyramid Fusion Network for Hyperspectral Pansharpening}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TNNLS.2023.3325887}}
