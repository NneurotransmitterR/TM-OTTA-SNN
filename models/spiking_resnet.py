# Based on: 
# https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/spiking_resnet.py
# https://github.com/yfguo91/MPBN/blob/main/models/resnet.py
# Modified ResNet models for MPBN
import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer, functional


__all__ = ['SpikingResNet_Cifar', 'SpikingResNet_M_Cifar', 'spiking_resnet19_m_cifar', 'spiking_resnet20_cifar', 'spiking_resnet19_cifar']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs), out_channels=planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs), out_channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs), out_channels=width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs), out_channels=width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs), out_channels=planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn3(out)

        return out

class SpikingResNet_Cifar(nn.Module):
    def __init__(self, block, layers, T=4, step_mode='s', num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None, inplanes=64, **kwargs):
        super(SpikingResNet_Cifar, self).__init__()
        if step_mode != 's':
            raise NotImplementedError("Multi-step mode is not supported yet!")
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        self.step_mode = step_mode
        self.T = T
        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False]
        if len(replace_stride_with_dilation) != 2:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 2-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs), out_channels=self.inplanes)
        self.layer1 = self._make_layer(block, 128, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)
        return x

    def forward(self, x):
        if self.step_mode == 's':
            out = []
            for _ in range(self.T):
                y = self._forward_impl(x)
                out.append(y)  # [N, num_classes] * T
            out = torch.stack(out, dim=0)
        elif self.step_mode == 'm':
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, N, C, H, W]
            out = self._forward_impl(x)  # [T, N, num_classes]
        return out


class SpikingResNet_M_Cifar(SpikingResNet_Cifar):
    def __init__(self, block, layers, T=4, step_mode='s', num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None, inplanes=64, **kwargs):
        super(SpikingResNet_M_Cifar, self).__init__(block, layers, T=T, step_mode=step_mode,
                                                   num_classes=num_classes, zero_init_residual=zero_init_residual,
                                                   groups=groups, width_per_group=width_per_group,
                                                   replace_stride_with_dilation=replace_stride_with_dilation,
                                                   norm_layer=norm_layer, spiking_neuron=spiking_neuron,
                                                   inplanes=inplanes, **kwargs)
        self.fc = layer.Linear(512 * block.expansion, 256, step_mode=step_mode)

        self.bn2 = layer.BatchNorm1d(256)
        self.sn2 = spiking_neuron(**deepcopy(kwargs), out_features=256)
        self.dropout = layer.Dropout(p=0.5)
        self.fc2 = layer.Linear(256, num_classes, step_mode=step_mode)
    
    def _forward_impl(self, x):
        x = super()._forward_impl(x)
        x = self.bn2(x)
        x = self.sn2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def _spiking_resnet_cifar(arch, block, layers, pretrained, progress, spiking_neuron, **kwargs):
    model = SpikingResNet_Cifar(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    return model


def spiking_resnet19_cifar(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    return _spiking_resnet_cifar('resnet19', BasicBlock, [3, 3, 2], pretrained, progress, spiking_neuron, **kwargs)


def spiking_resnet20_cifar(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    return _spiking_resnet_cifar('resnet20', BasicBlock, [3, 3, 3], pretrained, progress, spiking_neuron, **kwargs)


def _spiking_resnet_m(arch, block, layers, pretrained, progress, spiking_neuron, **kwargs):
    model = SpikingResNet_M_Cifar(block, layers, spiking_neuron=spiking_neuron, **kwargs)
    return model


def spiking_resnet19_m_cifar(pretrained=False, progress=True, spiking_neuron: callable=None, **kwargs):
    return _spiking_resnet_m('resnet19_m', BasicBlock, [3, 3, 2], pretrained, progress, spiking_neuron, **kwargs)


if __name__ == '__main__':
    from neurons import BNLIFNode
    sn19 = spiking_resnet19_cifar(spiking_neuron=BNLIFNode, T=4)
    sn20 = spiking_resnet20_cifar(spiking_neuron=BNLIFNode, T=4)
    sn19m = spiking_resnet19_m_cifar(spiking_neuron=BNLIFNode, T=4)
    print(sn19, sn20, sn19m)
    x = torch.rand([2, 3, 32, 32])
    out19 = sn19(x)
    out20 = sn20(x)
    out19m = sn19m(x)
    print(out19.shape, out20.shape, out19m.shape)  # [4, 1, 10]
    functional.reset_net(sn19)
    functional.reset_net(sn20)
    functional.reset_net(sn19m)
