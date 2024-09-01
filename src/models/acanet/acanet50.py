""" This file contains the class of ACANet, proposed by Apicella et al., in "Affordance segmentation of hand-occluded containers from exocentric images", IEEE/CVF ICCVW 2023.
    ACANet is an Affordance segmentation model that uses auxiliary branches to focus on the object and hand regions separately.
    The model learns affordance features under hand-occlusion by weighting the feature map through hand and object segmentation. """

import torch
import torch.nn as nn
import onnx_tool
import torch.nn.functional as F
import numpy as np

from torchvision import models


def convrelu(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ACANet(nn.Module):
    """ ACANet is a UNet-like convolutional neural network with a ResNet encoder.
        The decoder is composed of 3 branches:
        - one performs arm segmentation
        - one performs container segmentation,
        - one fuses the outputs of the other two branches with the features and performs arm and container affordances segmentation.
        """
    def __init__(self, n_class, pretrained=True, freeze_back=False):
        super().__init__()

        self.base_model = models.resnet50(pretrained)
        if freeze_back:
            for name, p in self.base_model.named_parameters():
                p.requires_grad = False
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]

        # Conv input channels derive from RN50
        # Conv output channels *2 compared to ACANet18
        self.conv_up3_0_aff = convrelu(1024 + 2048, 512, kernel_size=3, padding=1)
        self.conv_up3_0_obj = convrelu(1024 + 2048, 512, kernel_size=3, padding=1)
        self.conv_up3_0_hand = convrelu(1024 + 2048, 512, kernel_size=3, padding=1)
        self.conv_up3_1_aff = convrelu(512, 512, kernel_size=3, padding=1)
        self.conv_up3_1_obj = convrelu(512, 512, kernel_size=3, padding=1)
        self.conv_up3_1_hand = convrelu(512, 512, kernel_size=3, padding=1)

        self.conv_up2_0_aff = convrelu(512 + 512, 256, kernel_size=3, padding=1)
        self.conv_up2_0_obj = convrelu(512 + 512, 256, kernel_size=3, padding=1)
        self.conv_up2_0_hand = convrelu(512 + 512, 256, kernel_size=3, padding=1)
        self.conv_up2_1_aff = convrelu(256, 256, kernel_size=3, padding=1)
        self.conv_up2_1_obj = convrelu(256, 256, kernel_size=3, padding=1)
        self.conv_up2_1_hand = convrelu(256, 256, kernel_size=3, padding=1)

        self.conv_up1_0_aff = convrelu(256 + 256, 256, kernel_size=3, padding=1)
        self.conv_up1_0_obj = convrelu(256 + 256, 128, kernel_size=3, padding=1)
        self.conv_up1_0_hand = convrelu(256 + 256, 128, kernel_size=3, padding=1)
        self.conv_up1_1_aff = convrelu(256, 128, kernel_size=3, padding=1)
        self.conv_up1_1_obj = convrelu(128, 128, kernel_size=3, padding=1)
        self.conv_up1_1_hand = convrelu(128, 128, kernel_size=3, padding=1)
        self.conv_up1_2_aff = convrelu(128, 128, kernel_size=3, padding=1)

        self.conv_up0_0_aff = convrelu(128, 64, kernel_size=3, padding=1)
        self.conv_up0_0_obj = convrelu(128 + 64, 64, kernel_size=3, padding=1)
        self.conv_up0_0_hand = convrelu(128 + 64, 64, kernel_size=3, padding=1)
        self.conv_up0_1_aff = convrelu(64, 64, kernel_size=3, padding=1)
        self.conv_up0_1_obj = convrelu(64, 64, kernel_size=3, padding=1)
        self.conv_up0_1_hand = convrelu(64, 64, kernel_size=3, padding=1)

        self.conv_original_size_0_aff = convrelu(64, 32, kernel_size=3, padding=1)
        self.conv_original_size_0_obj = convrelu(64, 32, kernel_size=3, padding=1)
        self.conv_original_size_0_hand = convrelu(64, 32, kernel_size=3, padding=1)
        self.conv_original_size_1_aff = convrelu(32, 32, kernel_size=3, padding=1)
        self.conv_original_size_1_obj = convrelu(32, 32, kernel_size=3, padding=1)
        self.conv_original_size_1_hand = convrelu(32, 32, kernel_size=3, padding=1)

        self.conv_last_aff = nn.Conv2d(32, n_class, kernel_size=3, padding=1)
        self.conv_last_obj = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.conv_last_hand = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.projection_aff1 = nn.Conv2d(256, out_channels=256, kernel_size=1, padding=0)
        self.projection_aff2 = nn.Conv2d(256, out_channels=256, kernel_size=1, padding=0)

        self.weights_init(
            [self.conv_up3_0_aff, self.conv_up3_0_obj, self.conv_up3_0_hand, self.conv_up3_1_aff, self.conv_up3_1_obj,
             self.conv_up3_1_hand,
             self.conv_up2_0_aff, self.conv_up2_0_obj, self.conv_up2_0_hand, self.conv_up2_1_aff, self.conv_up2_1_obj,
             self.conv_up2_1_hand,
             self.conv_up1_0_aff, self.conv_up1_0_obj, self.conv_up1_0_hand, self.conv_up1_1_aff, self.conv_up1_1_obj,
             self.conv_up1_1_hand,
             self.conv_up0_0_aff, self.conv_up0_0_obj, self.conv_up0_0_hand, self.conv_up0_1_aff, self.conv_up0_1_obj,
             self.conv_up0_1_hand,
             self.conv_original_size_0_aff, self.conv_original_size_0_obj, self.conv_original_size_0_hand,
             self.conv_original_size_1_aff, self.conv_original_size_1_obj, self.conv_original_size_1_hand], "decoder")

        self.weights_init([self.conv_last_aff, self.conv_last_obj, self.conv_last_hand], "head")

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
        x_aff = self.conv_up3_0_aff(concat_1)
        x_obj = self.conv_up3_0_obj(concat_1)
        x_hand = self.conv_up3_0_hand(concat_1)
        x_aff = self.conv_up3_1_aff(x_aff)
        x_obj = self.conv_up3_1_obj(x_obj)
        x_hand = self.conv_up3_1_hand(x_hand)
        x_aff = F.interpolate(x_aff, scale_factor=2, mode="nearest")
        x_obj = F.interpolate(x_obj, scale_factor=2, mode="nearest")
        x_hand = F.interpolate(x_hand, scale_factor=2, mode="nearest")

        concat_2_aff = torch.cat([x_aff, layer2], dim=1)
        concat_2_obj = torch.cat([x_obj, layer2], dim=1)
        concat_2_hand = torch.cat([x_hand, layer2], dim=1)
        x_aff = self.conv_up2_0_aff(concat_2_aff)
        x_obj = self.conv_up2_0_obj(concat_2_obj)
        x_hand = self.conv_up2_0_hand(concat_2_hand)
        x_aff = self.conv_up2_1_aff(x_aff)
        x_obj = self.conv_up2_1_obj(x_obj)
        x_hand = self.conv_up2_1_hand(x_hand)
        x_aff = F.interpolate(x_aff, scale_factor=2, mode="nearest")
        x_obj = F.interpolate(x_obj, scale_factor=2, mode="nearest")
        x_hand = F.interpolate(x_hand, scale_factor=2, mode="nearest")

        concat_3_obj = torch.cat([x_obj, layer1], dim=1)
        concat_3_hand = torch.cat([x_hand, layer1], dim=1)
        x_obj = self.conv_up1_0_obj(concat_3_obj)
        x_hand = self.conv_up1_0_hand(concat_3_hand)
        x_obj = self.conv_up1_1_obj(x_obj)
        x_hand = self.conv_up1_1_hand(x_hand)
        x_obj = F.interpolate(x_obj, scale_factor=2, mode="nearest")
        x_hand = F.interpolate(x_hand, scale_factor=2, mode="nearest")

        concat_4_obj = torch.cat([x_obj, layer0], dim=1)
        concat_4_hand = torch.cat([x_hand, layer0], dim=1)
        x_obj = self.conv_up0_0_obj(concat_4_obj)
        x_hand = self.conv_up0_0_hand(concat_4_hand)
        x_obj = self.conv_up0_1_obj(x_obj)
        x_hand = self.conv_up0_1_hand(x_hand)
        x_obj = F.interpolate(x_obj, scale_factor=2, mode="nearest")
        x_hand = F.interpolate(x_hand, scale_factor=2, mode="nearest")

        x_obj = self.conv_original_size_0_obj(x_obj)
        x_hand = self.conv_original_size_0_hand(x_hand)
        x_obj = self.conv_original_size_1_obj(x_obj)
        x_hand = self.conv_original_size_1_hand(x_hand)
        out_obj = self.conv_last_obj(x_obj)
        out_hand = self.conv_last_hand(x_hand)

        out_hand_sig = torch.sigmoid(out_hand)
        out_hand_sig = F.interpolate(out_hand_sig, scale_factor=1 / 4, mode="bilinear", align_corners=True)
        out_obj_sig = torch.sigmoid(out_obj)
        out_obj_sig = F.interpolate(out_obj_sig, scale_factor=1 / 4, mode="bilinear", align_corners=True)

        proj_aff1 = self.projection_aff1(x_aff)
        proj_aff2 = self.projection_aff2(x_aff)
        x_aff1 = torch.mul(proj_aff1, out_hand_sig)
        x_aff2 = torch.mul(proj_aff2, out_obj_sig)
        x_aff_add = torch.add(x_aff1, x_aff2)
        x_aff_add = torch.add(x_aff_add, x_aff)

        concat_3_aff = torch.cat([x_aff_add, layer1], dim=1)
        x_aff = self.conv_up1_0_aff(concat_3_aff)
        x_aff = self.conv_up1_1_aff(x_aff)
        x_aff = self.conv_up1_2_aff(x_aff)
        x_aff = F.interpolate(x_aff, scale_factor=2, mode="nearest")

        x_aff = self.conv_up0_0_aff(x_aff)
        x_aff = self.conv_up0_1_aff(x_aff)
        x_aff = F.interpolate(x_aff, scale_factor=2, mode="nearest")

        x_aff = self.conv_original_size_0_aff(x_aff)
        x_aff = self.conv_original_size_1_aff(x_aff)
        out_aff = self.conv_last_aff(x_aff)
        return out_aff, out_obj, out_hand, concat_1, concat_2_aff, concat_2_obj, concat_3_obj, concat_2_hand, concat_3_hand


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES_NUM = 4
    model = ACANet(n_class=CLASSES_NUM, pretrained=True, freeze_back=False)
    model.to(device)

    # Save as ONNX file
    input_names = ['input_img']
    output_names = ['aff', 'obj', 'hand']
    x = torch.randn(1, 3, 480, 480, requires_grad=True).to(device)
    torch_out = model(x)
    tmp_file = 'acanet50.onnx'
    with torch.no_grad():
        torch_out = torch.onnx.export(model, x, tmp_file, input_names=input_names, output_names=output_names,
                                      opset_version=12)
        onnx_tool.model_profile(tmp_file)

    # From https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    # GPU WARM-UP
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
            curr_time = starter.elapsed_time(ender)  # milliseconds
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("Average time: ", mean_syn)
    print("Standard deviation time: ", std_syn)