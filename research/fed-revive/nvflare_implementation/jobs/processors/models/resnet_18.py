# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StatTracker(nn.Module):
    def __init__(self, num_channels, momentum=0.1):
        """
        Simple statistics tracker that mimics BatchNorm running statistics.
        Purely for model functionality, no synthesis-related state.

        Args:
            num_channels: C, number of feature channels
            momentum: for updating running_mean/var (like BatchNorm)
        """
        super().__init__()
        # Mimic BN.running_mean / running_var
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))
        self.momentum = momentum

    def forward(self, x, return_features=False):
        """
        Forward pass that optionally returns current batch statistics.

        Args:
            x: input tensor [B, C, H, W]
            return_features: if True, return (output, mean, var)

        Returns:
            output tensor, or (output, mean, var) if return_features=True
        """
        if return_features:
            # For synthesis: compute stats WITH gradients preserved
            mean = x.mean(dim=[0, 2, 3])  # → shape [C] (with gradients)
            var = x.var(dim=[0, 2, 3], unbiased=False)  # → shape [C] (with gradients)

            # Update running stats (detach ONLY for buffer updates, not return values)
            if self.training:
                with torch.no_grad():
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())

            # Return with gradients intact for synthesis loss computation
            return x, mean, var
        else:
            # For normal training: no gradient tracking for stats computation
            if self.training:
                with torch.no_grad():
                    # Compute stats only for running buffer updates, no gradients
                    mean = x.mean(dim=[0, 2, 3])
                    var = x.var(dim=[0, 2, 3], unbiased=False)
                    self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)

            return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.stat_track1 = StatTracker(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stat_track2 = StatTracker(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                        ),
                        ("stat_track", StatTracker(self.expansion * planes)),
                    ]
                )
            )

    def forward(self, x, return_features=False):
        if return_features:
            out, mean1, var1 = self.stat_track1(self.conv1(x), return_features=True)
            out = F.relu(out)
            out, mean2, var2 = self.stat_track2(self.conv2(out), return_features=True)

            # Check if shortcut has StatTracker (length > 1 means conv + StatTracker)
            if len(self.shortcut) > 1:
                shortcut_out, mean_sc, var_sc = self.shortcut.stat_track(self.shortcut.conv(x), return_features=True)
                features = [(mean1, var1), (mean2, var2), (mean_sc, var_sc)]
            else:
                shortcut_out = self.shortcut(x)
                features = [(mean1, var1), (mean2, var2)]

            out = F.relu(out + shortcut_out)
            return out, features
        else:
            out = self.stat_track1(self.conv1(x))
            out = F.relu(out)
            out = self.stat_track2(self.conv2(out))
            shortcut_out = self.shortcut(x)
            out = F.relu(out + shortcut_out)
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.block = block
        self.num_blocks = num_blocks

        # CIFAR-10 images are 32x32, so we use a smaller initial stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.stat_track1 = StatTracker(64)

        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * self.block.expansion, self.num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        if return_features:
            out, mean1, var1 = self.stat_track1(self.conv1(x), return_features=True)
            out = F.relu(out)

            # Collect features from all layers
            all_features = [(mean1, var1)]

            # Process each layer and collect features
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    out, block_features = block(out, return_features=True)
                    all_features.extend(block_features)

            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out, all_features
        else:
            out = self.stat_track1(self.conv1(x))
            out = F.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    def get_features(self, x):
        """
        Return the feature representation before the final linear layer.
        Useful for knowledge distillation.
        """
        out = self.stat_track1(self.conv1(x))
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        features = out.view(out.size(0), -1)
        return features


def reset_stat_tracker_stats(model):
    """
    Reset only the StatTracker running statistics in a model.
    Useful for synthesis or when you need to clear only StatTracker state.

    Args:
        model: PyTorch model containing StatTracker modules
    """
    for module in model.modules():
        if isinstance(module, StatTracker):
            # Reset running statistics to initial values
            module.running_mean.zero_()  # Reset to zeros
            module.running_var.fill_(1.0)  # Reset to ones
    return model


def ResNet18Local(num_classes=10, pretrained=False, pretrained_path=None, reset_stats=True):
    """
    Create a ResNet-18 model for local training with proper state management.

    This model is designed for client-side training in federated learning:
    - Includes StatTracker modules for tracking running statistics
    - Optionally resets running statistics for clean local training
    - Supports pretrained initialization

    Args:
        num_classes: Number of classes for the final layer
        pretrained: If True, loads pretrained weights
        pretrained_path: Path to pretrained model checkpoint
        reset_stats: If True, resets StatTracker running statistics (recommended for local training)

    Returns:
        ResNet18 model configured for local training
    """
    # Create model
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    model.has_running_stats = True  # Mark that this model has running stats

    # Load pretrained weights if requested
    if pretrained and pretrained_path is not None:
        if os.path.exists(pretrained_path):
            model.load_pretrained_weights(pretrained_path, strict=False)

    # Reset running statistics for clean local training
    if reset_stats:
        reset_stat_tracker_stats(model)

    # Set to training mode for local training
    model.train()

    return model


class ResNet18Wrapper(nn.Module):
    """
    Wrapper class to make ResNet18 serializable for NVFlare job export.
    This class stores only configuration (primitives) rather than class references,
    making it JSON-serializable by NVFlare's job export system.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # Create the actual model - this will be used at runtime
        # During NVFlare serialization, only primitive attributes (num_classes) are serialized
        self._model = None

    def _ensure_model(self):
        """Lazy initialization of the actual model."""
        if self._model is None:
            self._model = ResNet(BasicBlock, [2, 2, 2, 2], self.num_classes)
            self._model.has_running_stats = True

    def forward(self, x, return_features=False):
        self._ensure_model()
        return self._model(x, return_features=return_features)

    def state_dict(self, *args, **kwargs):
        self._ensure_model()
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self._ensure_model()
        return self._model.load_state_dict(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self._ensure_model()
        return self._model.train(mode)

    def eval(self):
        super().eval()
        self._ensure_model()
        return self._model.eval()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self._model is not None:
            self._model = self._model.to(*args, **kwargs)
        return self

    def modules(self):
        """Override modules() to return the wrapped model's modules."""
        self._ensure_model()
        return self._model.modules()

    def parameters(self, recurse=True):
        """Override parameters() to return the wrapped model's parameters."""
        self._ensure_model()
        return self._model.parameters(recurse=recurse)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        # Avoid recursion for special attributes
        if name in ["_model", "num_classes", "_parameters", "_buffers", "_modules"]:
            return super().__getattr__(name)

        # Ensure model exists before delegating
        try:
            self._ensure_model()
            return getattr(self._model, name)
        except AttributeError:
            return super().__getattr__(name)


def ResNet18Global(num_classes=10):
    """
    Create ResNet18 model with NVFlare serialization compatibility.

    This function returns a wrapper that's safe for NVFlare's job export system.
    The wrapper stores only primitive configuration values (num_classes) rather than
    class references, avoiding JSON serialization issues during job export.

    Args:
        num_classes: Number of output classes

    Returns:
        Resnet18Wrapper instance that behaves like ResNet18 but is serializable
    """
    return ResNet18Wrapper(num_classes=num_classes)
