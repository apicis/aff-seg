# Trained models for Semantic Object Affordances
This page contains models trained under the same framework and setup in "Segmenting Object Affordances: Reproducibility and Sensitivity to Scale" by Apicella et al. 

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)] 
[[code](https://github.com/apicis/aff-seg/)]


<!-- ## Data
* Hand-occluded object setting: [CORSMAL Hand-Occluded Containers Affordance (CHOC-AFF)](...) has $138,240$ images of 48 synthetic containers hand-held by synthetic hands rendered on top of 30 real backgrounds, and with annotations of the object affordances (\textit{graspable} and \textit{contain}) and forearm masks (\textit{arm}). 
The locations and poses of the hand-held objects vary across images, with 129,600 images of hand-held objects rendered above a table, and 8,640 images with objects rendered on a tabletop. CHOC-AFF has % splits the data in four sets 
four splits: a training set of 103,680 images, combining 26 backgrounds and 36 objects instances; a validation set of 17,280 images, using all 30 backgrounds and holding out 6 objects instances; a testing set of 13,824 images (CHOC-B), used to assess the generalisation performance to backgrounds never seen during training; and another testing set of 17,280 images (CHOC-I), to assess the generalisation performance to object instances never seen during training.

* Unoccluded object setting: [UMD](...) has 28,843 images of tools and containers, each placed on a blue rotating table, and acquired from a fixed view with the object in the center~\cite{myers2015affordance}. 
Object instances are not evenly distributed among the 17 object categories, e.g., there are 10 instances of spoon and 2 instances of pot. UMD is annotated with 7 affordance classes: \textit{grasp}, \textit{cut}, \textit{scoop}, \textit{contain}, \textit{pound}, \textit{support}, \textit{wrap-grasp}. UMD provides a pre-defined split of the dataset into training ($14,823$ images) and testing sets ($14,020$ images), holding out approximately half of object instances per category.


## Training setup
* Hand-occluded object setting: We use early stopping with a patience of $10$ epochs, setting the maximum number of epochs to $100$. The learning rate decreases by a factor of $0.5$ if there is no increase of the mean Intersection over Union in the validation set for $3$ consecutive epochs. We use the cropping window technique described in the previous work~\cite{apicella2023affordance}, resize images by a factor sampled in the interval $[1,1.5]$ and center crop the resized image with a $480 \times 480$ window. To simulate the other arm holding the object, we use horizontal flip with a probability of $0.5$. M2F-AFF uses batch size 4, learning rate $0.0001$, and an additional Gaussian noise augmentation with variance in range $[10, 100]$ to increase variability in training data.
* Unoccluded object setting: In this work, we train CNN and DRNAtt using cross-entropy loss, Adam optimiser and batch size $4$. For AffNet, we used a combination of cross-entropy and smooth-L1 losses for detection and cross-entropy for segmentation, mini-batch gradient descent with weight decay $0.001$ and batch size $2$ to fit our available GPU memory. We initialise all the mentioned architectures with a learning rate of $10^{-3}$. We train M2F-AFF with AdamW optimiser~\cite{loshchilov2017decoupled} with batch size $4$ and learning rate $10^{-4}$. We follow Mask2Former setup~\cite{cheng2022masked}, using hungarian algorithm~\cite{kuhn1955hungarian} to match the prediction from each latent vector with the corresponding annotation. We minimize the linear combination among the cross-entropy for classification $L_{cls}$, with the binary dice loss $L_{dice}$ and the binary cross-entropy $L_{ce}$ for segmentation. Losses are weighted using hyperparameters  $\lambda_{ce}$, $\lambda_{dice}$, and $\lambda_{cls}$. For all models, we decrease the learning rate by a factor of $0.5$, if there is no increase of the mean Intersection over Union in the validation set for $3$ consecutive epochs. We set the maximum number of epochs to $100$. We also use early stopping with a patience of $5$ epochs to reduce over-fitting. We use flipping with a probability of $0.5$, scaling by randomly sampling the scale factor in the interval $[1, 1.5]$ and center-cropping to simulate a zoom-in effect, color jitter with brightness, contrast, and saturation sampled randomly in the interval $[0.9, 1.1]$, and hue sampled randomly in the interval $[-0.1, 0.1]$. These augmentations allow to increase variability in the training set. We initialise all backbones with weights trained on ImageNet~\cite{deng2009imagenet}. -->


**Models date.** V1.0.0 - ... ... ... (Note: this is the date the model was trained.)

**Model type.** 
 
Models trained on hand-occluded object setting using [CHOC-AFF](...):
* [RN50-F](...): RN50-F uses a ResNet-50 encoder with a pyramid scene parsing module~\cite{zhao2017pyramid} to segment only the object affordances \textit{graspable} and \textit{contain}~\cite{hussain2020fpha}.  Both models implement a ResNet encoder~\cite{he2016deep}. 
* [ResNet18-UNet](...): RN18-U and ACANet~\cite{apicella2023affordance} are UNet-like models that gradually down-sample feature maps in the encoder and up-sample them in the decoder, preserving the information via skip connections~\cite{ronneberger2015u}.
* [ACANet](...): ACANet separately segments object and hand regions, using these masks to weigh the feature maps learnt in a third branch for the final affordance segmentation. Additionally, we trained a version of ACANet with ResNet-50.

Models trained on unoccluded object setting using [UMD](...):
* [AffordanceNet](...): AffordanceNet is a two-stage method that detects the object and segments affordances.
* [CNN](...): CNN is based on an encoder-decoder architecture to segment affordances. 

Models trained on both settings:
* [DRNAtt](...): DRNAtt uses position and channel attention mechanisms in parallel after the feature extraction stage~\cite{gu2021visual}. The outputs of the attention modules are summed element-wise and the result is up-sampled through a learnable decoder.
* [Mask2Former](...): Mask2Former is a recent hybrid architecture that combines an encoder-decoder convolutional neural network with a transformer decoder to decouple the classification of classes by the segmentation, tackling different types of segmentation, e.g., semantic, instance, and panoptic segmentation~\cite{cheng2022masked}. Mask2Former introduced a masking operation in the cross-attention mechanism that combines the latent vectors with the features extracted from the image, ignoring the pixel positions outside the object region. This type of processing, not considered by previous methods, can improve the learning in tasks such affordance segmentation, in which the majority of pixels belongs to the background.

**Training setup.** For ACANet, we use a linear combination of a Dice Loss for arm container segmentation branch, a binary cross-entropy loss with weight 1 for object segmentation, a binary cross-entropy loss with weight 1 for arm segmentation. We set the batch size to 2, the initial learning rate to 0.001, and we use the mini-batch Gradient Descent algorithm as optimizer with a momentum of 0.9 and a weight decay of 0.0001. We schedule the learning rate to decrease by a factor of 0.5, if there is no increase of the mean Intersection over Union in the validation set for 3 consecutive epochs. We use early stopping with a patience of 10 epochs to reduce overfitting, and set the maximum number of epochs to 100. We apply the following sequence of transformations: resize by a factor randomly sampled in the interval [1, 1.5] to avoid degrading quality; center crop the resized image with a W × H window to restore the original image resolution; and horizontal flip with a probability of 0.5 to simulate the other arm. We set the window size to W = H = 480.

**Citation details.**

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

**License.** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

**Enquiries, Question and Comments.** For enquiries, questions, or comments, please contact Tommaso Apicella.

**Primary intended uses.** The primary intended users of these trained models are academic researchers, scholars, and practitioners working in the fields of computer vision and robotics. The primary intended uses of models are:

* Assistive technologies for robotics and prosthetic applications (e.g., grasping, object manipulation) or collaborative human-robot scenarios (e.g., handovers).
* Baseline for affordance segmentation

**Out-of-scope use cases.** Any application which requires a high degree of accuracy and/or real-time requirements.

**Factors.** The model was trained on the extended version of CHOC dataset, which includes human forearm and hands that have textures from the SURREAL dataset. Note that these textures vary widely in skin tones. Backgrounds include both indoor and outdoor settings. Factors that may influence the performance are: cluttered background, lighting conditions, tablecloth with drawings, and textured clothing, object categories.

**Training Data.**
* **Datasets.** Mixed-reality training and validation sets from CORSMAL Hand-Occluded Containers (CHOC) dataset complemented with object affordances annotation.
* **Motivation.** Using mixed-reality datasets can easily scale the generation of a larger number of images under different realistic backgrounds, varying the hand and object poses.
* **Preprocessing.** RGB images are normalised in [0, 1] range, standardised using [0.485, 0.456, 0.406] per-channel mean and [0.229, 0.224, 0.225] per-channel standard deviation. Images can be of different resolutions and therefore we apply a cropping square window of fixed size to avoid distorsions or adding padding. Assuming a perfect object detector, we crop a W × W window around the center of the bounding box obtained from the object mask annotation to restrict the visual field and obtain an object centric view. However, the cropping window can go out of the support of the image if the bounding box is close to the image border. In this case, we extend the side of the window that is inside the image support to avoid padding. In case the bounding box is bigger than the cropping window, we crop the image inside the bounding box and resize it to the window size. W = 480 pixels.

**Evaluation Data.**
* **Datasets.** We evaluated the model in the following test sets:
  * Mixed-reality: 2 testing sets, one containing 13, 824 images, the other one 17, 280 images.
  * We sampled 150 images from CCM from the released training and public test set.
  * We sampled 150 images from HO-3D with the objects box and mug.
* **Motivation.**
  * Mixed reality: evaluate the models generalisation to different backgrounds and different object instances.
  * CCM and HO-3D: presence of various challenges, such as presence of the human body, real interactions, and different object instances and hand-object poses.
* **Preprocessing.** RGB images are are normalised in [0, 1] range, standardised using [0.485, 0.456, 0.406] per-channel mean and [0.229, 0.224, 0.225] per-channel standard deviation. We used the exact same training cropping procedure to evaluate the model on the mixed-reality testing sets. For CCM and HO-3D testing sets, we considered the visible object segmentation mask to recover the bounding box and consequently the W x W window.

**Metrics.**
* Model performance measures. Precision measures the percentage of true positives among all positive predicted pixels. Recall measures the percentage of true positive pixels with respect to the total number of positive pixels. The Jaccard Index measures how much two regions with the same support are comparable (Intersection over Union or IoU).
* Decision thresholds. The object and arm segmentation are rounded nearest, hence the output is 0 when the probability is less than 0.5, 1 when it is greater than 0.5 or equal 0.5.

**Quantitative Analyses.** Provided in the paper. ACANet achieves better affordance segmentation and generalisation than existing models.

**Ethical Considerations.** Even if the model is designed for assistive applications, the models were not tested in real use cases with humans involved. A proper analysis of the risks should be conducted before employing the models in such applications.
