# Segmenting Object Affordances: Reproducibility and Sensitivity to Scale

Visual affordance segmentation identifies image regions of an object an agent can interact with. 
Existing methods re-use and adapt learning-based architectures for semantic segmentation to the affordance segmentation task and evaluate on small-size datasets. 
However, experimental setups are often not reproducible, thus leading to unfair and inconsistent comparisons. 
In this work, we benchmark these methods under a reproducible setup on two single objects scenarios, tabletop without occlusions and hand-held containers, to facilitate future comparisons. 
We include a version of a recent architecture, Mask2Former, re-trained for affordance segmentation and show that this model is the best-performing on most testing sets of both scenarios. 
Our analysis show that models are not robust to scale variations when object resolutions differ from those in the training set.

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)]
[trained models]

---
## Table of Contents
1. [News](#news)
2. [Installation](#installation)
    1. [Setup specifics](#setup_specifics)  
    2. [Requirements](#requirements)
    3. [Instructions](#instructions)
3. [Running demo](#demo)
4. [Trained models](#trained_models)
5. [Training and testing data](#data)
   1. [Hand-occluded object setting](#handoccluded_data)
      <!-- 1. [Unoccluded object setting](#unoccluded_data)-->
6. [Contributing](#contributing)
7. [Credits](#credits)
8. [Enquiries, Question and Comments](#enquiries-question-and-comments)
9. [License](#license)

---

## News <a name="news"></a>
* ... September 2024: Released [code](src/models) and [weights]() of ACANet, ACANet50, RN18U, DRNAtt, RN50F, Mask2Former, trained on hand-occluded object setting ([CHOC-AFF](https://doi.org/10.5281/zenodo.5085800))
* 04 September 2024: Pre-print available on arxiv at https://arxiv.org/abs/2409.01814
* 17 August 2024: Source code, models, and further details will be released in the next weeks.
* 15 August 2024: Paper accepted at Twelfth International Workshop on Assistive Computer Vision and Robotics ([ACVR](https://iplab.dmi.unict.it/acvr2024/)), in conjunction with the 2024 European Conference on Computer Vision ([ECCV](https://eccv2024.ecva.net)).

---

## Installation <a name="installation"></a>

### Setup specifics <a name="setup_specifics"></a>
The models testing were performed using the following setup:
* *OS:* Ubuntu 18.04.6 LTS
* *Kernel version:* 4.15.0-213-generic
* *CPU:* Intel® Core™ i7-9700K CPU @ 3.60GHz
* *Cores:* 8
* *RAM:* 32 GB
* *GPU:* NVIDIA GeForce RTX 2080 Ti
* *Driver version:* 510.108.03
* *CUDA version:* 11.6

### Requirements <a name="requirements"></a> 
* Python 3.8
* PyTorch 1.9.0
* Torchvision 0.10.0
* OpenCV 4.10.0.84
* Numpy 1.24.4
* Tqdm 4.66.5

### Instructions <a name="instructions"></a>
```
# Create and activate conda environment
conda create -n affordance_segmentation python=3.8
conda activate affordance_segmentation
    
# Install libraries
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python onnx-tool numpy tqdm scipy
```

---
## Running demo <a name="demo"></a>

Download model checkpoint [ACANet.zip](https://doi.org/10.5281/zenodo.8364196), and unzip it.

Use the images in the folder *src/test_dir* or try with your own images. The folder structure is *DATA_DIR/rgb*. 

To run the model and visualise the output:

```
python src/demo.py --gpu_id=GPU_ID --model_name=MODEL_NAME --train_dataset=TRAIN_DATA --data_dir=DATA_DIR --checkpoint_path=CKPT_PATH --visualise_overlay=VIS_OVERLAY
```

* Replace *MODEL_NAME* with *acanet*
* *DATA_DIR*: directory where data are stored
* *TRAIN_DATA*: name of the training dataset
* *CKPT_PATH*: path to the .pth file
* *DEST_DIR*: path to the destination directory. This flag is considered only if you save the predictions ```--save_res=True``` or the overlay visualisation ```--save_overlay=True```. Results are automatically saved in *DEST_DIR/pred*, overlays in *DEST_DIR/vis*.

You can test if the model has the same performance by running inference on the images provided in *src/test_dir/rgb* and checking if the output is the same of *test_dir/pred* .

---
## Trained models <a name="trained_models"></a>
Here is the list of available models trained on UMD or CHOC-AFF

| Model name | UMD           | CHOC-AFF        |
|--------|---------------|-----------------|
| CNN | (Coming soon) |                 |
| AffordanceNet | (Coming soon)   |                 |
| ACANet |               | [link to zip](https://zenodo.org/records/8364197/files/ACANet.zip?download=1) |
| ACANet50 |               | link to zip     |
| RN50F |               | link to zip     |
| RN18U |               | link to zip     |
| DRNAtt | (Coming soon)   | link to zip     |
| Mask2Former | (Coming soon)   | link to zip     |

### Mask2Former installation
To use Mask2Former model, please run the following commands:
```
# Install detectron2 library
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Access mask2former folder in repository
cd src/models/mask2former

# Clone code from Mask2Former repository
git clone https://github.com/facebookresearch/Mask2Former.git 

# Compile 
cd Mask2Former/mask2former/modeling/pixel_decoder/ops
sh make.sh

# Return to the main directory (aff-seg)
cd ../../../../../../../../

# Install required libraries
pip install timm

# Run script to load Mask2Former (expected output: "Model loaded correctly!!")
python src/models/mask2former/test_mask2former_load.py
```
Comment out line 194 in */src/models/mask2former/Mask2Former/mask2former/maskformer_model.py* (`images = [(x - self.pixel_mean) / self.pixel_std for x in images]`) because we preprocess images in the dataloader.

### ResNet50FCN installation
To use ResNet50FastFCN (RN50F) model, please run the following commands:
```
# Access resnet_fcn folder in repository
cd src/models/resnet_fcn

# Clone code from FastFCN repository
git clone https://github.com/wuhuikai/FastFCN.git

# Return to the main directory (aff-seg)
cd ../../../
```
* In *aff-seg/src/models/resnet_fcn/FastFCN/encoding/models/encnet.py* replace line 11 `import encoding` with `from ..nn import encoding`
* In *aff-seg/src/models/resnet_fcn/FastFCN/encoding/models/base.py* replace in line 38 `pretrained=True` with `pretrained=False` (the script tries to download the resnet pretrained weights, but fails).
In case you want to use the pretrained weights, download them from [issue#86](https://github.com/wuhuikai/FastFCN/issues/86) and then modify line 27 `root='~/.encoding/models'` to point at the folder with the downloaded checkpoint.

Run script to load RN50F (expected output: model statistics, with average inference time and standard deviation):
```
python src/models/resnet_fcn/test_resnet_fcn_load.py
```

### DRNAtt installation
To use DRNAtt model, please run the following commands:
```
# Access drnatt folder in repository
cd src/models/drnatt

# Clone code from DANet repository
git clone https://github.com/junfu1115/DANet.git

# Clone code from DRN repository
git clone https://github.com/fyu/drn.git

# Return to the main directory (aff-seg)
cd ../../../
```
Comment out line 12 and 13 in */DANet/encoding/\__init__.py* (`from .version import __version__`, and `from . import nn, functions, parallel, utils, models, datasets, transforms`)

Run script to check that the model is correctly installed (expected output: model statistics, with average inference time and standard deviation):
```
python src/models/drnatt/drn_att.py
```


---
## Training and testing data <a name="data"></a>

<!-- ### Unoccluded object setting <a name="unoccluded_data"></a>
To recreate the training and testing splits:
1. Download [UMD Tools](https://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/) and unzip them in the preferred folder *SRC_DIR*.
2. Run ... to split data into training and testing sets following the object instances split introduced by [Myers et al.](https://users.umiacs.umd.edu/~fer/affordance/ICRA15_affordance_parts_final.pdf) -->


### Hand-occluded object setting <a name="handoccluded_data"></a>
To recreate the training and testing splits of the mixed-reality dataset:
1. Download [CHOC-AFF](https://doi.org/10.5281/zenodo.5085800) folders *rgb*, *mask*, *annotations*, *affordance* and unzip them in the preferred folder *SRC_DIR*. 
2. Run ```python src/utils/split_CHOC.py --src_dir=SRC_DIR --dst_dir=DST_DIR``` to split into training, validation and testing sets. *DST_DIR* is the directory where splits are saved.
3. Run ```python src/utils/create_dataset_crops_CHOC.py --data_dir=DATA_DIR --save=True --dest_dir=DEST_DIR``` to perform the cropping window procedure described in [ACANet paper](https://arxiv.org/abs/2308.11233). This script performs also the union between the arm mask and the affordance masks. *DATA_DIR* is the directory containing the *rgb* and *affordance* folders e.g.  *DST_DIR/training* following the naming used for the previous script. *DEST_DIR* is the destination directory, where to save cropped rgb images, and segmentation masks. 

To use the manually annotated data from [CCM](https://corsmal.eecs.qmul.ac.uk/containers_manip.html) and [HO-3D](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/) datasets: 
1. Download rgb and annotation files from [https://doi.org/10.5281/zenodo.10708553](https://doi.org/10.5281/zenodo.10708553) and unzip them in the preferred folder *SRC_DIR*. 
2. Run ```python src/utils/create_dataset_crops.py --data_dir=DATA_DIR --dataset_name=DATA_NAME --save=True --dest_dir=DEST_DIR``` to perform the cropping window procedure described in [ACANet paper](https://arxiv.org/abs/2308.11233). *DATA_DIR* is the directory containing the *rgb* and *affordance* folders. *DATA_NAME* is the dataset name (either CCM or HO3D). *DEST_DIR* is the destination directory, where to save cropped rgb images, and segmentation masks. 
---
## Contributing <a name="contributing"></a>

If you find an error, if you want to suggest a new feature or a change, you can use the issues tab to raise an issue with the appropriate label. 

Complete and full updates can be found in [CHANGELOG.md](CHANGELOG.md). The file follows the guidelines of [https://keepachangelog.com/en/1.1.0/](https://keepachangelog.com/en/1.1.0/).

---
## Credits <a name="credits"></a>

T. Apicella, A. Xompero, P. Gastaldo, A. Cavallaro, <i>Segmenting Object Affordances: Reproducibility and Sensitivity to Scale</i>, 
Proceedings of the European Conference on Computer Vision Workshops, Twelfth International Workshop on Assistive Computer Vision and Robotics (ACVR),
Milan, Italy, 29 September 2024.

```
@InProceedings{Apicella2024ACVR_ECCVW,
            title = {Segmenting Object Affordances: Reproducibility and Sensitivity to Scale},
            author = {Apicella, T. and Xompero, A. and Gastaldo, P. and Cavallaro, A.},
            booktitle = {Proceedings of the European Conference on Computer Vision Workshops},
            note = {Twelfth International Workshop on Assistive Computer Vision and Robotics},
            address={Milan, Italy},
            month="29" # SEP,
            year = {2024},
        }
```

---

## Enquiries, Question and Comments <a name="enquiries-question-and-comments"></a>

If you have any further enquiries, question, or comments, or you would like to file a bug report or a feature request, please use the Github issue tracker. 

---

## Licence <a name="license"></a>

This work is licensed under the MIT License. To view a copy of this license, see
[LICENSE](LICENSE).
