## This is a pytorch implementation of the paper *Domain-Adverserial Training of Neural Network*

#### Dataset

First, you need download two datasets: source dataset mnist,

```
cd dataset
mkdir mnist
cd mnist
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

and target dataset mnist_m from [pan.baidu.com](https://pan.baidu.com/s/1eShdX0u) or [Google Drive](https://drive.google.com/open?id=0B_tExHiYS-0veklUZHFYT19KYjg)

```
cd dataset
mkdir mnist_m
cd mnist_m
tar -zvxf mnist_m.tar.gz
```

#### Training

Then, run `main.py`

