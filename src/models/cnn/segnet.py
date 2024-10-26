""" Re-implementation of SegNet based on 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation' by Badrinarayanan et al.
    The code is inspired by https://github.com/vinceecws/SegNet_PyTorch/blob/master/Pavements/SegNet.py """

import torch
import torch.nn as nn
import numpy as np

from torchvision import models


def convrelu(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SegNet(nn.Module):
    """ Segnet architecture. SegNet is an encoder-decoder architecture for semantic segmentation.
    The encoder extracts the features from the image and the decoder maps the extracted features back to the image resolution.
    The features down-sampling is not performed by the encoder blocks (VGG16) but by down-sampling layers and the indices
    of down-sampled position are saved to be re-used in the decoder.
    """

    def __init__(self, n_class, pretrained=True, freeze_back=False):
        super(SegNet, self).__init__()

        self.base_model = models.vgg16_bn(pretrained)
        if freeze_back:
            for name, p in self.base_model.named_parameters():
                p.requires_grad = False
        self.base_layers = list(self.base_model.children())[0]
        del self.base_model

        # Encoder
        self.maxpool_enc = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.layer1 = nn.Sequential(*self.base_layers[0:6])
        self.layer2 = nn.Sequential(*self.base_layers[7:13])
        self.layer3 = nn.Sequential(*self.base_layers[14:23])
        self.layer4 = nn.Sequential(*self.base_layers[24:33])
        self.layer5 = nn.Sequential(*self.base_layers[34:43])
        del self.base_layers

        # Decoder
        self.maxpool_dec = nn.MaxUnpool2d(2, stride=2)

        self.convrelu5_1 = convrelu(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.convrelu5_2 = convrelu(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.convrelu5_3 = convrelu(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.convrelu4_1 = convrelu(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.convrelu4_2 = convrelu(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.convrelu4_3 = convrelu(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.convrelu3_1 = convrelu(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.convrelu3_2 = convrelu(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.convrelu3_3 = convrelu(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.convrelu2_1 = convrelu(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.convrelu2_2 = convrelu(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.convrelu1_1 = convrelu(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.convrelu1_2 = nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=3, padding=1)

    def forward(self, x):
        # ENCODER LAYERS
        # Stage 1
        x = self.layer1(x)
        x, ind1 = self.maxpool_enc(x)
        size1 = x.size()

        # Stage 2
        x = self.layer2(x)
        x, ind2 = self.maxpool_enc(x)
        size2 = x.size()

        # Stage 3
        x = self.layer3(x)
        x, ind3 = self.maxpool_enc(x)
        size3 = x.size()

        # Stage 4
        x = self.layer4(x)
        x, ind4 = self.maxpool_enc(x)
        size4 = x.size()

        # Stage 5
        x = self.layer5(x)
        x, ind5 = self.maxpool_enc(x)
        size5 = x.size()

        # DECODER LAYERS
        # Stage 5
        x = self.maxpool_dec(x, ind5, output_size=size4)
        x = self.convrelu5_3(self.convrelu5_2(self.convrelu5_1(x)))

        # Stage 4
        x = self.maxpool_dec(x, ind4, output_size=size3)
        x = self.convrelu4_3(self.convrelu4_2(self.convrelu4_1(x)))

        # Stage 3
        x = self.maxpool_dec(x, ind3, output_size=size2)
        x = self.convrelu3_3(self.convrelu3_2(self.convrelu3_1(x)))

        # Stage 2
        x = self.maxpool_dec(x, ind2, output_size=size1)
        x = self.convrelu2_2(self.convrelu2_1(x))

        # Stage 1
        x = self.maxpool_dec(x, ind1)
        out = self.convrelu1_2(self.convrelu1_1(x))

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES_NUM = 4
    model = SegNet(n_class=CLASSES_NUM, pretrained=True, freeze_back=False)
    model.to(device)

    x = torch.randn(1, 3, 480, 480).to(device)

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
            curr_time = starter.elapsed_time(ender)  # milliseconds
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print("Average time: ", mean_syn)
    print("Standard deviation time: ", std_syn)
