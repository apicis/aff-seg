# Mask2Former-AFF

Mask2Former decouples the mask segmentation and mask classification tackling different types of segmentation, e.g., semantic, instance, and panoptic. Mask2Former is designed by [Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation", CVPR, 2022](https://arxiv.org/abs/2112.01527). In the following, the details of our setup implementation.

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)] 
[[code](https://github.com/apicis/aff-seg/)]
[[umd data](https://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/)]

**Model date.** V1.0.0 - 22 June 2024 (Note: this is the date the model was trained.)

**Model type.** Mask2Former is a hybrid architecture that combines an encoder-decoder convolutional neural network with a transformer decoder to decouple the classification of classes by the segmentation. Mask2Former introduced a masking operation in the cross-attention mechanism that combines the latent vectors with the features extracted from the image, ignoring the pixel positions outside the object region. This type of processing, not considered by previous methods, can improve the learning in tasks such affordance segmentation, in which the majority of pixels belongs to the background.

**Training setup.** For Mask2Former, we use a linear combination among the cross-entropy for classification, the binary dice loss and the binary cross-entropy for segmentation, with hungarian algorithm to match the prediction from each latent vector with the corresponding annotation. We set the weight for cross-entropy to 5, for dice to 5, for binary cross-entropy to 2, we sample sets of K=12,544 points for prediction and ground truth masks. We set the batch size to 4, the initial learning rate to 0.0001, and we use the AdamW algorithm as optimizer with default hyperparameters. We schedule the learning rate to decrease by a factor of 0.5, if there is no increase of the mean Intersection over Union in the validation set for 3 consecutive epochs. We use early stopping with a patience of 5 epochs to reduce overfitting, and set the maximum number of epochs to 100. We apply the following sequence of transformations: resize by a factor randomly sampled in the interval [1, 1.5] to avoid degrading quality; center crop the resized image with a W × H window to restore the original image resolution; horizontal flip with a probability of 0.5; color jitter with brightness, contrast, and saturation sampled randomly in the interval [0.9, 1.1], and hue sampled randomly in the interval [-0.1, 0.1]. We set the window size to W = 640, H = 480.

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

**Factors.** The model was trained on the [UMD dataset](https://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/), which includes a single object on a blue rotating tabletop. Factors that may influence the performance are: cluttered background, lighting conditions, table colors, object instances.

**Training Data.**

* Datasets. Training and validation set from the [UMD dataset](https://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/). UMD training set has 14,423 images of tools and containers, each placed on a blue rotating table, and acquired from a fixed view with the object in the center. The training set is split into training and validation sets, sampling $30\%$ of the images for the latter.
* Motivation. Train models on an unoccluded object scenario using different object instances. 
* Preprocessing. RGB images are normalised in [0, 1] range, standardised using [0.485, 0.456, 0.406] per-channel mean and [0.229, 0.224, 0.225] per-channel standard deviation. 

**Evaluation Data.**

* Datasets. We evaluated the model on UMD testing set having 14,020 images and object instances unseen during training.
* Motivation. Evaluate the models generalisation to different object instances.
* Preprocessing. RGB images are normalised in [0, 1] range, standardised using [0.485, 0.456, 0.406] per-channel mean and [0.229, 0.224, 0.225] per-channel standard deviation.

**Metrics.**

* Model performance measures. The Jaccard Index measures how much two regions with the same support are comparable (Intersection over Union or IoU).
* Decision thresholds. During inference, we use a confidence threshold of $0.5$ to separate background from an affordance region.

**Quantitative Analyses.** Provided in the paper. Mask2Former achieves better affordance segmentation and generalisation than existing models on most testing sets.

**Ethical Considerations.** Even if the model is designed for assistive applications, the model was not tested in real use cases with humans involved. A proper analysis of the risks should be conducted before employing the model in such applications.
