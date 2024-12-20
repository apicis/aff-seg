""" This script tests that ResNet50FastFCN-PSP is correctly loaded. Follow README.md instructions before running this script. """
import torch
import onnx_tool
import numpy as np

from models.resnet_fcn.FastFCN.encoding.models import get_segmentation_model

def load_resnet_fcn(n_classes):
    head_name = 'psp'
    backbone = 'resnet50'
    jpu = 'JPU'
    pretrained = False
    dilated = False
    aux = True
    model = get_segmentation_model(head_name,
                                   backbone=backbone,
                                   dilated=dilated,
                                   jpu=jpu,
                                   num_classes=n_classes,
                                   aux=aux,
                                   pretrained=pretrained)
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CLASSES_NUM = 3
    model = load_resnet_fcn(CLASSES_NUM)
    model.to(device)
    model.eval()

    # Save as ONNX file
    input_names = ['input_img']
    output_names = ['aff']
    x = torch.randn(1, 3, 480, 480).to(device)
    tmp_file = 'resnet_fcn.onnx'
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
