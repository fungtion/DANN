## This is a pytorch implementation of the paper *[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)*


#### Environment
- Pytorch 1.0
- Python 2.7

#### Network Structure


![p8KTyD.md.jpg](https://s1.ax1x.com/2018/01/12/p8KTyD.md.jpg)

#### Dataset

First, you need download the target dataset mnist_m from [pan.baidu.com](https://pan.baidu.com/s/1pXaMkVsQf_yUT51SeYh27g) fetch code: kjan or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg)

```
cd dataset
mkdir mnist_m
cd mnist_m
tar -zvxf mnist_m.tar.gz
```

#### Training

Then, run `main.py`

python 3 and docker version please go to [DANN_py3](https://github.com/fungtion/DANN_py3) 

