# DRNAtt

DRNAtt is designed by [Gu et al., "Visual affordance detection using an efficient attention convolutional neural network", *Neurocomputing*, 2021](https://www.sciencedirect.com/science/article/pii/S0925231221000278). In the following, the details of our re-implementation.

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)] 
[[code](https://github.com/apicis/aff-seg/)]
[[umd data](https://users.umiacs.umd.edu/~fer/affordance/part-affordance-dataset/)]

**Model date.** V1.0.0 - 23 June 2024 (Note: this is the date the model was trained.)

**Model type.** DRNAtt is an encoder-decoder network for affordance segmentation. DRNAtt uses position and channel attention mechanisms in parallel after the feature extraction stage. The outputs of the attention modules are summed element-wise and the result is up-sampled through a learnable decoder. DRNAtt uses Dilated Residual Network (DRN) backbone to maintain higher resolution compared to a ResNet backbone.

**Training setup.** To train DRNAtt, we use a cross-entropy loss for the affordance segmentation. We set the batch size to 4, the initial learning rate to 0.001, and we use the Adam algorithm as optimizer with default configuration (betas=(0.9, 0.999)). We schedule the learning rate to decrease by a factor of 0.5, if there is no increase of the mean Intersection over Union in the validation set for 3 consecutive epochs. 
We use early stopping with a patience of 5 epochs to reduce overfitting, and set the maximum number of epochs to 100. We apply the following sequence of transformations: resize by a factor randomly sampled in the interval [1, 1.5] to avoid degrading quality; center crop the resized image with a W × H window to restore the original image resolution; horizontal flip with a probability of 0.5; color jitter with brightness, contrast, and saturation sampled randomly in the interval [0.9, 1.1], and hue sampled randomly in the interval [-0.1, 0.1]. We set the window size to W = 640, H = 480.


**Citation details.**

T. Apicella, A. Xompero, P. Gastaldo, A. Cavallaro, <i>Segmenting Object Affordances: Reproducibility and Sensitivity to Scale</i>, 
Proceedings of the European Conference on Computer Vision Workshops, Twelfth International Workshop on Assistive Computer Vision and Robotics (ACVR), Milan, Italy, 29 September 2024.

```
@inproceedings{apicella2023affordance,
  title={Affordance segmentation of hand-occluded containers from exocentric images},
  author={Apicella, Tommaso and Xompero, Alessio and Ragusa, Edoardo and Berta, Riccardo and Cavallaro, Andrea and Gastaldo, Paolo},
  booktitle={IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year={2023},
}
```

**License.** Creative Commons Attribution 4.0 International

**Enquiries, Question and Comments.** For enquiries, questions, or comments, please contact Tommaso Apicella.

**Primary intended uses.** The primary intended users of this model are academic researchers, scholars, and practitioners working in the fields of computer vision and robotics. The primary intended uses of DRNAtt are:

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

**Quantitative Analyses.** Provided in the paper. DRNAtt is the second best model on UMD testing set.

**Ethical Considerations.** Even if the model is designed for assistive applications, the model was not tested in real use cases with humans involved. A proper analysis of the risks should be conducted before employing the model in such applications.
