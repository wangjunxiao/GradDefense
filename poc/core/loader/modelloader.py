"""Define basic models and translate some torchvision stuff."""
"""Stuff from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py."""
import torch
import torchvision

from .network.convnet import ConvNet
from .network.lenet import LeNet
from .network.srnet import SRNet
from .network.resnet import ResNet
from .network.revnet import iRevNet
from .network.densenet import _DenseNet, _Bottleneck

from collections import OrderedDict
import numpy as np
import core.utils as utils


def load_model(model, seed=None, num_classes=10, num_channels=3):
    """Return various models."""
    if seed is None:
        model_init_seed = np.random.randint(0, 2**32 - 10, dtype=np.int64)
    else:
        model_init_seed = seed
    utils.set_random_seed(model_init_seed)

    if model == 'ConvNet64': # MNIST CIFAR10 CIFAR100
        model = ConvNet(width=64, num_channels=num_channels, num_classes=num_classes)
    elif model == 'ConvNet4096': # MNIST CIFAR10 CIFAR100
        model = ConvNet(width=4096, num_channels=num_channels, num_classes=num_classes)
    elif model == 'ConvNet8':
        model = ConvNet(width=8, num_channels=num_channels, num_classes=num_classes)
    elif model == 'ConvNet16':
        model = ConvNet(width=16, num_channels=num_channels, num_classes=num_classes)
    elif model == 'ConvNet32':
        model = ConvNet(width=32, num_channels=num_channels, num_classes=num_classes)
    elif model == 'BeyondInferringMNIST':
        model = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)),
            ('relu0', torch.nn.LeakyReLU()),
            ('conv2', torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            ('relu1', torch.nn.LeakyReLU()),
            ('conv3', torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            ('relu2', torch.nn.LeakyReLU()),
            ('conv4', torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('relu3', torch.nn.LeakyReLU()),
            ('flatt', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(12544, 12544)),
            ('relu4', torch.nn.LeakyReLU()),
            ('linear1', torch.nn.Linear(12544, 10)),
            ('softmax', torch.nn.Softmax(dim=1))
        ]))
    elif model == 'BeyondInferringCifar':
        model = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)),
            ('relu0', torch.nn.LeakyReLU()),
            ('conv2', torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            ('relu1', torch.nn.LeakyReLU()),
            ('conv3', torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            ('relu2', torch.nn.LeakyReLU()),
            ('conv4', torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('relu3', torch.nn.LeakyReLU()),
            ('flatt', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(12544, 12544)),
            ('relu4', torch.nn.LeakyReLU()),
            ('linear1', torch.nn.Linear(12544, 10)),
            ('softmax', torch.nn.Softmax(dim=1))
        ]))
    elif model == 'MLP':
        width = 1024
        model = torch.nn.Sequential(OrderedDict([
            ('flatten', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(3072, width)),
            ('relu0', torch.nn.ReLU()),
            ('linear1', torch.nn.Linear(width, width)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(width, width)),
            ('relu2', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(width, num_classes))]))
    elif model == 'TwoLP':
        width = 2048
        model = torch.nn.Sequential(OrderedDict([
            ('flatten', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(3072, width)),
            ('relu0', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(width, num_classes))]))
    elif model == 'ResNet20':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16)
    elif model == 'ResNet20-nostride':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16,
                       strides=[1, 1, 1, 1])
    elif model == 'ResNet20-10':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet20-4':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 4)
    elif model == 'ResNet20-4-unpooled':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 4,
                       pool='max')
    elif model == 'ResNet28-10':
        model = ResNet(torchvision.models.resnet.BasicBlock, [4, 4, 4], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet32':
        model = ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16)
    elif model == 'ResNet32-10':
        model = ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet44':
        model = ResNet(torchvision.models.resnet.BasicBlock, [7, 7, 7], num_classes=num_classes, base_width=16)
    elif model == 'ResNet56':
        model = ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], num_classes=num_classes, base_width=16)
    elif model == 'ResNet110':
        model = ResNet(torchvision.models.resnet.BasicBlock, [18, 18, 18], num_classes=num_classes, base_width=16)
    elif model == 'ResNet18':
        model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64)
    elif model == 'ResNet34':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet50':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet50-2':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64 * 2)
    elif model == 'ResNet101':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet152':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, base_width=64)
    elif model == 'MobileNet':
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # cifar adaptation, cf.https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        model = torchvision.models.MobileNetV2(num_classes=num_classes,
                                               inverted_residual_setting=inverted_residual_setting,
                                               width_mult=1.0)
        model.features[0] = torchvision.models.mobilenet.ConvBNReLU(num_channels, 32, stride=1)  # this is fixed to width=1
    elif model == 'MNASNet':
        model = torchvision.models.MNASNet(1.0, num_classes=num_classes, dropout=0.2)
    elif model == 'DenseNet121':
        model = torchvision.models.DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                                            num_init_features=64, bn_size=4, drop_rate=0, num_classes=num_classes,
                                            memory_efficient=False)
    elif model == 'DenseNet40':
        model = _DenseNet(_Bottleneck, [6, 6, 6, 0], growth_rate=12, num_classes=num_classes)
    elif model == 'DenseNet40-4':
        model = _DenseNet(_Bottleneck, [6, 6, 6, 0], growth_rate=12 * 4, num_classes=num_classes)
    elif model == 'SRNet3':
        model = SRNet(upscale_factor=3, num_channels=num_channels)
    elif model == 'SRNet1':
        model = SRNet(upscale_factor=1, num_channels=num_channels)
    elif model == 'iRevNet':
        if num_classes <= 100:
            in_shape = [num_channels, 32, 32]  # only for cifar right now
            model = iRevNet(nBlocks=[18, 18, 18], nStrides=[1, 2, 2],
                            nChannels=[16, 64, 256], nClasses=num_classes,
                            init_ds=0, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=4)
        else:
            in_shape = [3, 224, 224]  # only for imagenet
            model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                            nChannels=[24, 96, 384, 1536], nClasses=num_classes,
                            init_ds=2, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=4)
    elif model == 'LeNet':
        model = LeNet(num_channels=num_channels, num_classes=num_classes)
    else:
        raise NotImplementedError('Model not implemented.')

    print(f'Model initialized with random key {model_init_seed}.')
    
    return model, model_init_seed

