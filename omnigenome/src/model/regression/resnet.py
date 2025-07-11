# -*- coding: utf-8 -*-
# file: resnet.py
# time: 14:43 29/01/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
# Adapted from: https://github.com/terry-r123/RNABenchmark/blob/main/downstream/structure/resnet.py
"""
ResNet implementation for genomic sequence analysis.

This module provides a ResNet architecture adapted for processing genomic sequences
and their structural representations. It includes basic blocks, bottleneck blocks,
and a complete ResNet implementation optimized for genomic data.
"""
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, List, Optional


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for the convolution (default: 1)
        groups (int): Number of groups for grouped convolution (default: 1)
        dilation (int): Dilation factor for the convolution (default: 1)
        
    Returns:
        nn.Conv2d: 3x3 convolution layer
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for the convolution (default: 1)
        
    Returns:
        nn.Conv2d: 1x1 convolution layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    5x5 convolution with padding.
    
    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int): Stride for the convolution (default: 1)
        groups (int): Number of groups for grouped convolution (default: 1)
        dilation (int): Dilation factor for the convolution (default: 1)
        
    Returns:
        nn.Conv2d: 5x5 convolution layer
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=2,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    """
    Basic ResNet block for genomic sequence processing.
    
    This block implements a basic residual connection with two convolutions
    and is optimized for processing genomic sequence data with layer normalization.
    
    Attributes:
        expansion (int): Expansion factor for the block (default: 1)
        conv1: First 3x3 convolution layer
        bn1: First layer normalization
        conv2: Second 5x5 convolution layer
        bn2: Second layer normalization
        relu: ReLU activation function
        drop: Dropout layer
        downsample: Downsampling layer for residual connection
        stride: Stride for the convolutions
    """
    
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        # base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Initialize the BasicBlock.
        
        Args:
            inplanes (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for the convolutions (default: 1)
            downsample: Downsampling layer for residual connection (default: None)
            groups (int): Number of groups for grouped convolution (default: 1)
            dilation (int): Dilation factor for convolutions (default: 1)
            norm_layer: Normalization layer type (default: None, uses LayerNorm)
            
        Raises:
            NotImplementedError: If dilation > 1 is specified
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        # if groups != 1 or base_width != 64:
        # raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.drop = nn.Dropout(0.25, inplace=False)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the BasicBlock.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tensor: Output tensor with same shape as input
        """
        identity = x

        x = x.permute(0, 2, 3, 1)
        out = self.bn1(x)
        out = out.permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv1(out)

        out = out.permute(0, 2, 3, 1)
        out = self.bn2(out)
        out = out.permute(0, 3, 1, 2)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck ResNet block for genomic sequence processing.
    
    This block implements a bottleneck residual connection with three convolutions
    (1x1, 3x3, 1x1) and is designed for deeper networks. It's adapted from
    the original ResNet V1.5 implementation.
    
    Attributes:
        expansion (int): Expansion factor for the block (default: 4)
        conv1: First 1x1 convolution layer
        bn1: First batch normalization
        conv2: Second 3x3 convolution layer
        bn2: Second batch normalization
        conv3: Third 1x1 convolution layer
        bn3: Third batch normalization
        relu: ReLU activation function
        downsample: Downsampling layer for residual connection
        stride: Stride for the convolutions
    """
    
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Initialize the Bottleneck block.
        
        Args:
            inplanes (int): Number of input channels
            planes (int): Number of output channels
            stride (int): Stride for the convolutions (default: 1)
            downsample: Downsampling layer for residual connection (default: None)
            groups (int): Number of groups for grouped convolution (default: 1)
            base_width (int): Base width for the bottleneck (default: 64)
            dilation (int): Dilation factor for convolutions (default: 1)
            norm_layer: Normalization layer type (default: None, uses BatchNorm2d)
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the Bottleneck block.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tensor: Output tensor with same shape as input
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture adapted for genomic sequence analysis.
    
    This ResNet implementation is specifically designed for processing genomic
    sequences and their structural representations. It uses layer normalization
    instead of batch normalization and is optimized for genomic data characteristics.
    
    Attributes:
        _norm_layer: Normalization layer type
        inplanes: Number of input channels for the first layer
        dilation: Dilation factor for convolutions
        groups: Number of groups for grouped convolutions
        base_width: Base width for bottleneck blocks
        conv1: Initial convolution layer
        bn1: Initial normalization layer
        relu: ReLU activation function
        layer1: First layer of ResNet blocks
        fc1: Final fully connected layer
    """

    def __init__(
        self,
        channels,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 1,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ) -> None:
        """
        Initialize the ResNet architecture.
        
        Args:
            channels (int): Number of input channels
            block: Type of ResNet block (BasicBlock or Bottleneck)
            layers (List[int]): List specifying the number of blocks in each layer
            zero_init_residual (bool): Whether to zero-initialize residual connections (default: False)
            groups (int): Number of groups for grouped convolutions (default: 1)
            width_per_group (int): Width per group for bottleneck blocks (default: 1)
            replace_stride_with_dilation: Whether to replace stride with dilation (default: None)
            norm_layer: Normalization layer type (default: None, uses LayerNorm)
            
        Raises:
            ValueError: If replace_stride_with_dilation is not None or a 3-element tuple
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self._norm_layer = norm_layer

        self.inplanes = 48
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            channels, self.inplanes, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 48, layers[0])
        self.fc1 = nn.Linear(48, 1)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """
        Create a layer of ResNet blocks.
        
        Args:
            block: Type of ResNet block to use
            planes (int): Number of output channels for the layer
            blocks (int): Number of blocks in the layer
            stride (int): Stride for the first block (default: 1)
            dilate (bool): Whether to use dilation (default: False)
            
        Returns:
            nn.Sequential: Sequential container of ResNet blocks
        """
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
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        Forward pass implementation.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tensor: Output tensor after processing through ResNet
        """
        # [bz,hd,len,len]
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.bn1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(x)

        x = self.layer1(x)
        x = x.mean(dim=[2, 3])
        x = self.fc1(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ResNet.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tensor: Output tensor after processing through ResNet
        """
        return self._forward_impl(x)


def resnet_b16(channels=128, bbn=16):
    """
    Create a ResNet-B16 model for genomic sequence analysis.
    
    This function creates a ResNet model with 16 basic blocks, optimized
    for processing genomic sequences and their structural representations.
    
    Args:
        channels (int): Number of input channels (default: 128)
        bbn (int): Number of basic blocks (default: 16)
        
    Returns:
        ResNet: Configured ResNet model
    """
    return ResNet(channels, BasicBlock, [bbn])
