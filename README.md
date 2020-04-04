# When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks 
This repository contains the implementation code for paper [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) (__CVPR2020__). Also see the [project page](http://www.mit.edu/~yuzhe/robnets.html).

In this work, we take an architectural perspective and investigate the patterns of network architectures that are resilient to adversarial attacks. We discover a family of robust architectures (__RobNets__), which exhibit superior robustness performance to other widely used architectures.

## Installation

### Prerequisites
-__Data__ Download the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/) and [ImageNet](http://image-net.org/download) dataset and move the test/validation set to the folder `data/`.

-__Model__ Download the [pre-trained models](https://drive.google.com/file/d/1h2JLcumQgS296Su950ZEtiJrEgxWzxfP/view?usp=sharing) and unziped to the folder `checkpoint`.


### Dependencies for RobNets 
You can install the dependencies for RobNets using
```bash
pip install -r requirements.txt
```


## Experiments
All the configurations of the experiments are provided in the folders `experiments/*/config.py`, including different datasets and RobNet architectures. You can directly modify them to suit you demand.

To conduct a specific experiment, e.g. `RobNet_free` for CIFAR10, run
```
python main.py --config='./experiments/RobNet_free_cifar10/config.py'
```

## Acknowledgements
The implementation of RobNets is partly based on [this work](https://github.com/quark0/darts).


## Citation
If you find the idea or code useful for your research, please cite [our paper](https://arxiv.org/abs/1911.10695):
```bib
@article{guo2019meets,
  title={When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks},
  author={Guo, Minghao and Yang, Yuzhe and Xu, Rui and Liu, Ziwei and Lin, Dahua},
  journal={arXiv preprint arXiv:1911.10695},
  year={2019}
}
```
