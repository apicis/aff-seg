# Segmenting Object Affordances: Reproducibility and Sensitivity to Scale

Visual affordance segmentation identifies image regions of an object an agent can interact with. 
Existing methods re-use and adapt learning-based architectures for semantic segmentation to the affordance segmentation task and evaluate on small-size datasets. 
However, experimental setups are often not reproducible, thus leading to unfair and inconsistent comparisons. 
In this work, we benchmark these methods under a reproducible setup on two single objects scenarios, tabletop without occlusions and hand-held containers, to facilitate future comparisons. 
We include a version of a recent architecture, Mask2Former, re-trained for affordance segmentation and show that this model is the best-performing on most testing sets of both scenarios. 
Our analysis show that models are not robust to scale variations when object resolutions differ from those in the training set.

[arXiv]
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
4. [Training and testing data](#data) 
   1. [Unoccluded object setting](#unoccluded_data)
   2. [Hand-occluded object setting](#handoccluded_data)
5. [Contributing](#contributing)
6. [Credits](#credits)
7. [Enquiries, Question and Comments](#enquiries-question-and-comments)
8. [License](#license)

---

## News <a name="news"></a>
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
* Python ...
* PyTorch ...
* Torchvision ...
* OpenCV ...
* Numpy ...
* Tqdm ...

### Instructions <a name="instructions"></a>
```
# Create and activate conda environment
conda create -n affordance_segmentation python=3.8
conda activate affordance_segmentation
    
# Install libraries
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install opencv-python onnx-tool numpy tqdm 
```

---
## Running demo <a name="demo"></a>

Download model checkpoint [ACANet.zip](https://doi.org/10.5281/zenodo.8364196), and unzip it.

Use the images in the folder *test_dir* or try with your own images. The folder structure is *DATA_DIR/rgb*. 

To run the model and visualise the output:

```
python3 demo.py --model_name=MODEL_NAME --data_dir=DATA_DIR  --checkpoint_path=CHECKPOINT_PATH --visualise_overlay=True --dest_dir=DEST_DIR
```

* Replace *MODEL_NAME* with *acanet*
* *DATA_DIR*: directory where data are stored
* *CHECKPOINT_PATH*: path to the .pth file
* *DEST_DIR*: path to the destination directory. This flag is considered only if you save the predictions ```--save_res=True``` or the overlay visualisation ```--save_overlay=True```. Results are automatically saved in *DEST_DIR/pred*, overlays in *DEST_DIR/vis*.

You can test if the model has the same performance by running inference on the images provided in *test_dir/rgb* and checking if the output is the same of *test_dir/pred* .

---
## Training and testing data <a name="data"></a>

### Unoccluded object setting <a name="unoccluded_data"></a>


### Hand-occluded object setting <a name="handoccluded_data"></a>
To recreate the training and testing splits of the mixed-reality dataset:
1. Download the [dataset](https://doi.org/10.5281/zenodo.5085800) folders *rgb*, *mask*, *annotations*, *affordance* and unzip them in the preferred folder *SRC_DIR*. 
2. Run ```utils/split_dataset.py --src_dir=SRC_DIR --dst_dir=DST_DIR``` to split into training, validation and testing sets. *DST_DIR* is the directory where splits are saved.
3. Run ```utils/create_dataset_crops.py --data_dir=DATA_DIR --save=True --dest_dir=DEST_DIR``` to perform the cropping window procedure described in [ACANet paper](). This script performs also the union between the arm mask and the affordance masks. *DATA_DIR* is the directory containing the *rgb* and *affordance* folders e.g.  *DST_DIR/training* following the naming used for the previous script. *DEST_DIR* is the destination directory, where to save cropped rgb images, and segmentation masks. 

To use the manually annotated data from [CCM](https://corsmal.eecs.qmul.ac.uk/containers_manip.html) and [HO-3D](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/) datasets: 
1. Download files at [https://doi.org/10.5281/zenodo.10708553](https://doi.org/10.5281/zenodo.10708553)
2. 

---
## Contributing

If you find an error, if you want to suggest a new feature or a change, you can use the issues tab to raise an issue with the appropriate label. 

Complete and full updates can be found in [CHANGELOG.md](CHANGELOG.md). The file follows the guidelines of [https://keepachangelog.com/en/1.1.0/](https://keepachangelog.com/en/1.1.0/).

---
## Credits

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

## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, or you would like to file a bug report or a feature request, please use the Github issue tracker. 

---

## Licence

This work is licensed under the MIT License.  To view a copy of this license, see
[LICENSE](LICENSE).