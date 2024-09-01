""" This file contains the class of ResNet18Unet.
    The code is modified from https://www.kaggle.com/code/ateplyuk/pytorch-starter-u-net-resnet (Apache License, Version 2.0)
    The main changes to the code:
    * Different class name
    * Different convolutional layers
    * Weights initialisation function
    * Main function to test that the model is running
    """

import torch
import torch.nn as nn
import onnx_tool
import torch.nn.functional as F
import numpy as np

# from torchsummary import summary
from torchvision import models


def convrelu(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResNet18Unet(nn.Module):
    """ ResNet18Unet is a UNet-like convolutional neural network with a ResNet18 encoder.  Consist of *encoder*
    and *decoder* parts connected by *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections."""
    def __init__(self, n_class, pretrained=True, freeze_back=False):
        super().__init__()

        self.base_model = models.resnet18(pretrained)
        if freeze_back:
            for name, p in self.base_model.named_parameters():
                p.requires_grad = False
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        self.conv_up3_0 = convrelu(256 + 512, 256, kernel_size=3, padding=1)
        self.conv_up3_1 = convrelu(256, 256, kernel_size=3, padding=1)
        self.conv_up2_0 = convrelu(128 + 256, 128, kernel_size=3, padding=1)
        self.conv_up2_1 = convrelu(128, 128, kernel_size=3, padding=1)
        self.conv_up1_0 = convrelu(64 + 128, 64, kernel_size=3, padding=1)
        self.conv_up1_1 = convrelu(64, 64, kernel_size=3, padding=1)
        self.conv_up0_0 = convrelu(64 + 64, 32, kernel_size=3, padding=1)
        self.conv_up0_1 = convrelu(32, 32, kernel_size=3, padding=1)
        self.conv_original_size_0 = convrelu(32, 16, kernel_size=3, padding=1)
        self.conv_original_size_1 = convrelu(16, 16, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(16, n_class, kernel_size=3, padding=1)

        self.weights_init([self.conv_up3_0, self.conv_up3_1,
                           self.conv_up2_0, self.conv_up2_1,
                           self.conv_up1_0, self.conv_up1_1,
                           self.conv_up0_0, self.conv_up0_1,
                           self.conv_original_size_0,
                           self.conv_original_size_1], "decoder")

        self.weights_init([self.conv_last], "head")

    def weights_init(self, list_of_layers, module="decoder"):
        """Initialisation of decoder and head layers. """
        if module == "decoder":
            for m in list_of_layers:
                for layer in m:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)

                    elif isinstance(layer, nn.BatchNorm2d):
                        nn.init.constant_(layer.weight, 1)
                        nn.init.constant_(layer.bias, 0)

                    elif isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if m.bias is not None:
                            nn.init.constant_(layer.bias, 0)

        elif module == "head":
            for m in list_of_layers:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # Backbone
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = F.interpolate(layer4, scale_factor=2, mode="nearest")

        concat_1 = torch.cat([x, layer3], dim=1)
        x = self.conv_up3_0(concat_1)
        x = self.conv_up3_1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        concat_2 = torch.cat([x, layer2], dim=1)
        x = self.conv_up2_0(concat_2)
        x = self.conv_up2_1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        concat_3 = torch.cat([x, layer1], dim=1)
        x = self.conv_up1_0(concat_3)
        x = self.conv_up1_1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        concat_4 = torch.cat([x, layer0], dim=1)
        x = self.conv_up0_0(concat_4)
        x = self.conv_up0_1(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        x = self.conv_original_size_0(x)
        x = self.conv_original_size_1(x)
        out = self.conv_last(x)

        return out, concat_1, concat_2, concat_3, concat_4


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES_NUM = 4
    model = ResNet18Unet(n_class=CLASSES_NUM, pretrained=True, freeze_back=False)
    model.to(device)

    # Save as ONNX file
    input_names = ['input_img']
    output_names = ['aff']
    x = torch.randn(1, 3, 480, 480).to(device)
    tmp_file = 'resnet_unet.onnx'
    with torch.no_grad():
        torch_out = torch.onnx.export(model, x, tmp_file, input_names=input_names, output_names=output_names,
                                      opset_version=12)
        onnx_tool.model_profile(tmp_file)

    # From https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(30):
        _ = model(x)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(x)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # milliseconds
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("Average time: ", mean_syn)
    print("Standard deviation time: ", std_syn)
