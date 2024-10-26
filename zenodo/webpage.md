# Trained models for Semantic Object Affordances
This page contains models trained in "Segmenting Object Affordances: Reproducibility and Sensitivity to Scale" by Apicella et al. 

Compared to previous works, models are trained under the same setup on two single objects scenarios, tabletop without occlusions and hand-held containers, to facilitate future comparisons. 

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)] 
[[code](https://github.com/apicis/aff-seg/)]

## Release notes
26 October 2024:
* Upload weights of models trained on [UMD](https://ieeexplore.ieee.org/document/7139369)
* Model cards are in the zenodo folder of the [code repository](https://github.com/apicis/aff-seg/tree/main/zenodo/model_cards/umd)

26 September 2024:
* Upload weights of models trained on [CHOC-AFF](https://arxiv.org/abs/2308.11233)
* Model cards are in the zenodo folder of the [code repository](https://github.com/apicis/aff-seg/tree/main/zenodo/model_cards/choc-aff) 

## Available models
Models trained on hand-occluded object setting using [CHOC-AFF](https://arxiv.org/abs/2308.11233):
* [RN50-F](https://ieeexplore.ieee.org/document/9190733): RN50-F uses a ResNet-50 encoder with a pyramid scene parsing module to segment only the object affordances *graspable* and *contain*. 
* [ResNet18-UNet](https://arxiv.org/abs/1505.04597): UNet-like model that gradually down-sample feature maps in the encoder and up-sample them in the decoder, preserving the information via skip connections.
* [ACANet](https://arxiv.org/abs/2308.11233): ACANet separately segments object and hand regions, using these masks to weigh the feature maps learnt in a third branch for the final affordance segmentation. We trained also a version of ACANet with ResNet-50.

Models trained on unoccluded object setting using [UMD](https://ieeexplore.ieee.org/document/7139369):
* [AffordanceNet](https://arxiv.org/abs/1709.07326): AffordanceNet is a two-stage method that detects the object and segments affordances.
* [CNN](https://ieeexplore.ieee.org/document/7759429): CNN is based on an encoder-decoder architecture to segment affordances. 

Models trained on both settings:
* [DRNAtt](https://www.sciencedirect.com/science/article/pii/S0925231221000278): DRNAtt uses position and channel attention mechanisms in parallel after the feature extraction stage. The outputs of the attention modules are summed element-wise and the result is up-sampled through a learnable decoder.
* [Mask2Former](https://arxiv.org/abs/2112.01527): Mask2Former is a recent hybrid architecture that combines an encoder-decoder convolutional neural network with a transformer decoder to decouple the classification of classes by the segmentation, tackling different types of segmentation, e.g., semantic, instance, and panoptic segmentation. Mask2Former introduced a masking operation in the cross-attention mechanism that combines the latent vectors with the features extracted from the image, ignoring the pixel positions outside the object region.

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

**Enquiries, Question and Comments.** For enquiries, questions, or comments, please contact [Tommaso Apicella](mailto:tommaso.apicella@edu.unige.it) or open a issue in the GitHub repository.
