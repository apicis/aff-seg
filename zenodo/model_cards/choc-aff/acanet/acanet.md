# ACANet

ACANet is an affordance segmentation model designed by [Apicella et al., "Affordance segmentation of hand-occluded containers from exocentric images", ICCVW, 2023](https://arxiv.org/abs/2308.11233v1).

[[arXiv](https://arxiv.org/abs/2308.11233v1)] 
[[webpage](https://apicis.github.io/projects/acanet.html)]
[[code](https://github.com/SEAlab-unige/acanet)]
[[mixed-reality data](https://doi.org/10.5281/zenodo.5085800)]
[[real testing data](https://doi.org/10.5281/zenodo.10708553)]

**Model date.** V1.0.0 - 27 May 2023 (Note: this is the date the model was trained.)

**Model type.** ACANet is a UNet-like convolutional neural network with a ResNet encoder. The decoder is composed of 3 branches: one performs arm segmentation, one performs container segmentation, one fuses the outputs of the other two branches with the features and performs arm and container affordances segmentation.

**Training setup.** For ACANet, we use a linear combination of a Dice Loss for arm container segmentation branch, a binary cross-entropy loss with weight 1 for object segmentation, a binary cross-entropy loss with weight 1 for arm segmentation. We set the batch size to 2, the initial learning rate to 0.001, and we use the mini-batch Gradient Descent algorithm as optimizer with a momentum of 0.9 and a weight decay of 0.0001. We schedule the learning rate to decrease by a factor of 0.5, if there is no increase of the mean Intersection over Union in the validation set for 3 consecutive epochs. We use early stopping with a patience of 10 epochs to reduce overfitting, and set the maximum number of epochs to 100. We apply the following sequence of transformations: resize by a factor randomly sampled in the interval [1, 1.5] to avoid degrading quality; center crop the resized image with a W × H window to restore the original image resolution; and horizontal flip with a probability of 0.5 to simulate the other arm. We set the window size to W = H = 480.

**Citation details.**

Affordance segmentation of hand-occluded containers from exocentric images
T. Apicella, A. Xompero, E. Ragusa, R. Berta, A. Cavallaro, P. Gastaldo
IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 2023

```
@inproceedings{apicella2023affordance,
  title={Affordance segmentation of hand-occluded containers from exocentric images},
  author={Apicella, Tommaso and Xompero, Alessio and Ragusa, Edoardo and Berta, Riccardo and Cavallaro, Andrea and Gastaldo, Paolo},
  booktitle={IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2023},
}
```

**License.** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

**Enquiries, Question and Comments.** For enquiries, questions, or comments, please contact Tommaso Apicella.

**Primary intended uses.** The primary intended users of this model are academic researchers, scholars, and practitioners working in the fields of computer vision and robotics. The primary intended uses of ACANet are:

* Assistive technologies for robotics and prosthetic applications (e.g., grasping, object manipulation) or collaborative human-robot scenarios (e.g., handovers).
* Baseline for affordance segmentation

**Out-of-scope use cases.** Any application which requires a high degree of accuracy and/or real-time requirements.

**Factors.** The model was trained on the extended version of [CHOC dataset](https://doi.org/10.5281/zenodo.8332421), which includes human forearm and hands that have textures from the SURREAL dataset. Note that these textures vary widely in skin tones. Backgrounds include both indoor and outdoor settings. Factors that may influence the performance are: cluttered background, lighting conditions, tablecloth with drawings, and textured clothing, object categories.

**Training Data.**

* Datasets. Mixed-reality training and validation sets from CORSMAL Hand-Occluded Containers ([CHOC](https://doi.org/10.5281/zenodo.8332421)) dataset complemented with object affordances annotation.
* Motivation. Using mixed-reality datasets can easily scale the generation of a larger number of images under different realistic backgrounds, varying the hand and object poses.
* Preprocessing. RGB images are normalised in [0, 1] range, standardised using [0.485, 0.456, 0.406] per-channel mean and [0.229, 0.224, 0.225] per-channel standard deviation. Images can be of different resolutions and therefore we apply a cropping square window of fixed size to avoid distorsions or adding padding. Assuming a perfect object detector, we crop a W × W window around the center of the bounding box obtained from the object mask annotation to restrict the visual field and obtain an object centric view. However, the cropping window can go out of the support of the image if the bounding box is close to the image border. In this case, we extend the side of the window that is inside the image support to avoid padding. In case the bounding box is bigger than the cropping window, we crop the image inside the bounding box and resize it to the window size. W = 480 pixels.

**Evaluation Data.**

* Datasets. We evaluated the model in the following test sets:
    - Mixed-reality: 2 testing sets, one containing 13, 824 images, the other one 17, 280 images.
    - We sampled 150 images from CCM from the released training and public test set.
    - We sampled 150 images from HO-3D with the objects box and mug.
* Motivation.
    - Mixed reality: evaluate the models generalisation to different backgrounds and different object instances.
    - CCM and HO-3D: presence of various challenges, such as presence of the human body, real interactions, and different object instances and hand-object poses.
* Preprocessing. RGB images are are normalised in [0, 1] range, standardised using [0.485, 0.456, 0.406] per-channel mean and [0.229, 0.224, 0.225] per-channel standard deviation. We used the exact same training cropping procedure to evaluate the model on the mixed-reality testing sets. For CCM and HO-3D testing sets, we considered the visible object segmentation mask to recover the bounding box and consequently the W x W window.

**Metrics.**

* Model performance measures. Precision measures the percentage of true positives among all positive predicted pixels. Recall measures the percentage of true positive pixels with respect to the total number of positive pixels. The Jaccard Index measures how much two regions with the same support are comparable (Intersection over Union or IoU).
* Decision thresholds. The object and arm segmentation are rounded nearest, hence the output is 0 when the probability is less than 0.5, 1 when it is greater than 0.5 or equal 0.5.

**Quantitative Analyses.** Provided in the paper. ACANet achieves better affordance segmentation and generalisation than existing models.

**Ethical Considerations.** Even if the model is designed for assistive applications, the model was not tested in real use cases with humans involved. A proper analysis of the risks should be conducted before employing the model in such applications.
