# This file is modified from official pycls repository

"""Model and loss construction functions."""

from pycls.core.net import SoftCrossEntropyLoss
from pycls.models.resnet import *
from pycls.models.vgg import *
from pycls.models.alexnet import *

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
# Supported models
_models = {
    # VGG style architectures
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,

    # ResNet style archiectures
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,

    # AlexNet architecture
    'alexnet': alexnet
}

# Supported loss functions
_loss_funs = {"cross_entropy": SoftCrossEntropyLoss}


class FeaturesNet(nn.Module):
    def __init__(self, in_layers, out_layers, use_mlp=False, penultimate_active=False):
        super().__init__()
        self.use_mlp = use_mlp
        self.penultimate_active = penultimate_active
        self.lin1 = nn.Linear(in_layers, in_layers)
        self.lin2 = nn.Linear(in_layers, in_layers)
        self.final = nn.Linear(in_layers, out_layers)

    def forward(self, x):
        feats = x
        if self.use_mlp:
            x = F.relu(self.lin1(x))
            x = F.relu((self.lin2(x)))
        out = self.final(x)
        if self.penultimate_active:
            return feats, out
        return out


def get_model(cfg):
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun(cfg):
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model(cfg):
    """Builds the model."""
    if cfg.MODEL.LINEAR_FROM_FEATURES:
        num_features = 384 if cfg.DATASET.NAME in ['IMAGENET50', 'IMAGENET100', 'IMAGENET200'] else 512
        return FeaturesNet(num_features, cfg.MODEL.NUM_CLASSES)

    model = get_model(cfg)(num_classes=cfg.MODEL.NUM_CLASSES, use_dropout=True)
    if cfg.DATASET.NAME == 'MNIST':
        model.conv1 =  torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    return model 


def build_loss_fun(cfg):
    """Build the loss function."""
    return get_loss_fun(cfg)()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor


class Pip_resnet18(nn.Module):
    def __init__(self, resnet, num_nb, num_lms=68, input_size=256, net_stride=32):
        super(Pip_resnet18, self).__init__()
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.sigmoid = nn.Sigmoid()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.net_stride == 128:
            self.layer5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.layer6 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn6 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

            nn.init.normal_(self.layer6.weight, std=0.001)
            if self.layer6.bias is not None:
                nn.init.constant_(self.layer6.bias, 0)
            nn.init.constant_(self.bn6.weight, 1)
            nn.init.constant_(self.bn6.bias, 0)
        elif self.net_stride == 64:
            self.layer5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)
        elif self.net_stride == 32:
            pass
        elif self.net_stride == 16:
            self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
            self.bn_deconv1 = nn.BatchNorm2d(512)
            nn.init.normal_(self.deconv1.weight, std=0.001)
            if self.deconv1.bias is not None:
                nn.init.constant_(self.deconv1.bias, 0)
            nn.init.constant_(self.bn_deconv1.weight, 1)
            nn.init.constant_(self.bn_deconv1.bias, 0)
        else:
            print('No such net_stride!')
            exit(0)

        self.cls_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.x_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.y_layer = nn.Conv2d(512, num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_x_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)
        self.nb_y_layer = nn.Conv2d(512, num_nb*num_lms, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))
        elif self.net_stride == 16:
            x = F.relu(self.bn_deconv1(self.deconv1(x)))
        else:
            pass
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5


def get_keypoint_model(num_nb=10, num_lms=21, input_size=256, net_stride=32):
    bb_model = torchvision.models.resnet18()
    model = Pip_resnet18(bb_model, num_nb, num_lms=num_lms, input_size=input_size, net_stride=net_stride)
    return model

