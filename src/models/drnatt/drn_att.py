""" Re-implementation of DRNAtt from "Visual affordance detection using an efficient attention convolutional neural network" by Gu et al.
    Read instructions in README.md before running this script.
"""
import torch
import torch.nn as nn
import onnx_tool
import numpy as np

from models.drnatt.DANet.encoding.nn.da_att import PAM_Module, CAM_Module
from models.drnatt.drn.drn import drn_d_22


class DRNatt(nn.Module):
    def __init__(self, n_class, pretrained=True):
        super().__init__()

        self.base_model = drn_d_22(pretrained=pretrained)
        self.layer0 = self.base_model.layer0
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.layer5 = self.base_model.layer5
        self.layer6 = self.base_model.layer6
        self.layer7 = self.base_model.layer7
        self.layer8 = self.base_model.layer8

        self.sa = PAM_Module(512)
        self.sc = CAM_Module(512)

        self.upsampling = nn.Sequential(
            nn.Conv2d(512, 8 * 8 * n_class, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(8),
        )  # from https://github.com/Lornatang/ESPCN-PyTorch/blob/master/model.py

    def forward(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        sa_feat = self.sa(x)
        sc_feat = self.sc(x)
        feat_sum = sa_feat + sc_feat

        out_aff = self.upsampling(feat_sum)

        return out_aff


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES_NUM = 4
    model = DRNatt(n_class=CLASSES_NUM, pretrained=False)
    model.to(device)

    # Save as ONNX file
    input_names = ['input_img']
    output_names = ['aff']
    x = torch.randn(1, 3, 480, 480, requires_grad=True).to(device)
    torch_out = model(x)
    tmp_file = 'drn_att.onnx'
    with torch.no_grad():
        torch_out = torch.onnx.export(model, x, tmp_file, input_names=input_names, output_names=output_names,
                                      opset_version=12)
        onnx_tool.model_profile(tmp_file)  # comment pixel shuffle in upsampling layer

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
