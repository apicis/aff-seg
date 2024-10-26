# Mask2Former-AFF

Mask2Former decouples the mask segmentation and mask classification tackling different types of segmentation, e.g., semantic, instance, and panoptic. Mask2Former is designed by [Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation", CVPR, 2022](https://arxiv.org/abs/2112.01527). In the following, the details of our setup implementation.

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)] 
[[code](https://github.com/apicis/aff-seg/)]
[[mixed-reality data](https://doi.org/10.5281/zenodo.5085800)]
[[real testing data](https://doi.org/10.5281/zenodo.10708553)]

**Model date.** V1.0.0 - 07 July 2024 (Note: this is the date the model was trained.)

**Model type.** Mask2Former is a hybrid architecture that combines an encoder-decoder convolutional neural network with a transformer decoder to decouple the classification of classes by the segmentation. Mask2Former introduced a masking operation in the cross-attention mechanism that combines the latent vectors with the features extracted from the image, ignoring the pixel positions outside the object region. This type of processing, not considered by previous methods, can improve the learning in tasks such affordance segmentation, in which the majority of pixels belongs to the background.

**Training setup.** For Mask2Former, we use a linear combination among the cross-entropy for classification, the binary dice loss and the binary cross-entropy for segmentation, with hungarian algorithm to match the prediction from each latent vector with the corresponding annotation. We set the weight for cross-entropy to 5, for dice to 5, for binary cross-entropy to 2, we sample sets of K=12,544 points for prediction and ground truth masks. We set the batch size to 4, the initial learning rate to 0.0001, and we use the AdamW algorithm as optimizer with default hyperparameters. We schedule the learning rate to decrease by a factor of 0.5, if there is no increase of the mean Intersection over Union in the validation set for 3 consecutive epochs. We use early stopping with a patience of 10 epochs to reduce overfitting, and set the maximum number of epochs to 100. We apply the following sequence of transformations: resize by a factor randomly sampled in the interval [1, 1.5] to avoid degrading quality; center crop the resized image with a W × H window to restore the original image resolution; horizontal flip with a probability of 0.5 to simulate the other arm; and Gaussian noise augmentation with variance in range $[10, 100]$. We set the window size to W = H = 480.

**Citation details.**

T. Apicella, A. Xompero, P. Gastaldo, A. Cavallaro, <i>Segmenting Object Affordances: Reproducibility and Sensitivity to Scale</i>, 
Proceedings of the European Conference on Computer Vision Workshops, Twelfth International Workshop on Assistive Computer Vision and Robotics (ACVR), Milan, Italy, 29 September 2024.

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

**License.** Creative Commons Attribution 4.0 International

**Enquiries, Question and Comments.** For enquiries, questions, or comments, please contact Tommaso Apicella.

**Primary intended uses.** The primary intended users of this model are academic researchers, scholars, and practitioners working in the fields of computer vision and robotics. The primary intended uses of Mask2Former-AFF are:

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
* Decision thresholds. During inference, we use a confidence threshold of $0.5$ to separate background from an affordance region.

**Quantitative Analyses.** Provided in the paper. Mask2Former achieves better affordance segmentation and generalisation than existing models on most testing sets.

**Ethical Considerations.** Even if the model is designed for assistive applications, the model was not tested in real use cases with humans involved. A proper analysis of the risks should be conducted before employing the model in such applications.
