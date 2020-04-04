# When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks 
This repository contains the implementation code for paper [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) (__CVPR 2020__). Also check out the [project page](http://www.mit.edu/~yuzhe/robnets.html).

In this work, we take an architectural perspective and investigate the patterns of network architectures that are resilient to adversarial attacks. We discover a family of robust architectures (__RobNets__), which exhibit superior robustness performance to other widely used architectures.

![overview](assets/robnets.png)

## Installation

### Prerequisites
- __Data:__ Download the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [SVHN](http://ufldl.stanford.edu/housenumbers/) and [ImageNet](http://image-net.org/download) dataset and move the test/validation set to the folder `data/`.

- __Model:__ Download the [pre-trained models](https://drive.google.com/file/d/1h2JLcumQgS296Su950ZEtiJrEgxWzxfP/view?usp=sharing) and unzip to the folder `checkpoint/`.


### Dependencies for RobNets 
You can install the dependencies for RobNets using
```bash
pip install -r requirements.txt
```


## Experiments
All the configurations of the experiments are provided in folders `experiments/*/config.py`, including different datasets and RobNet architectures. You can directly modify them to suit your demand.

To conduct a specific experiment, e.g. `RobNet_free` for CIFAR10, run
```bash
python main.py --config='./experiments/RobNet_free_cifar10/config.py'
```


## Use RobNet Architectures
To use the searched RobNet models, for example, load `RobNet_free` on CIFAR10:
```python
import models
import architecture_code

# use RobNet architecture
net = models.robnet(architecture_code.robnet_free)
net = net.cuda()
# load pre-trained model
net.load_state_dict(torch.load('./checkpoint/RobNet_free_cifar10.pth.tar'))
```
For other models, the loading process is similar, just copy the corresponding parameters (you can find in the variable `model_param` in each `experiments/*/config.py`) to the function `models.robnet()`.


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


## Contact
Please contact guomh2014@gmail.com and yuzheyangpku@gmail.com if you have any questions. Enjoy!
