# Trained models for Semantic Object Affordances
This page contains models trained in "Segmenting Object Affordances: Reproducibility and Sensitivity to Scale" by Apicella et al. 

Compared to previous works, models are trained under the same setup on two single objects scenarios, tabletop without occlusions and hand-held containers, to facilitate future comparisons. 

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)] 
[[code](https://github.com/apicis/aff-seg/)]

## Release notes
... September 2024:
* Upload weights of models trained on [CHOC-AFF](https://arxiv.org/abs/2308.11233)


## Available models
Models trained on hand-occluded object setting using [CHOC-AFF](https://arxiv.org/abs/2308.11233):
* [RN50-F](https://arxiv.org/abs/1903.11816): RN50-F uses a ResNet-50 encoder with a pyramid scene parsing module to segment only the object affordances *graspable* and *contain*. 
* [ResNet18-UNet](https://arxiv.org/abs/1505.04597): UNet-like model that gradually down-sample feature maps in the encoder and up-sample them in the decoder, preserving the information via skip connections.
* [ACANet](https://arxiv.org/abs/2308.11233): ACANet separately segments object and hand regions, using these masks to weigh the feature maps learnt in a third branch for the final affordance segmentation. We trained also a version of ACANet with ResNet-50.
<!-- Models trained on unoccluded object setting using [UMD](...):
* [AffordanceNet](...): AffordanceNet is a two-stage method that detects the object and segments affordances.
* [CNN](...): CNN is based on an encoder-decoder architecture to segment affordances. 
Models trained on both settings: -->
* [DRNAtt](https://www.sciencedirect.com/science/article/pii/S0925231221000278): DRNAtt uses position and channel attention mechanisms in parallel after the feature extraction stage. The outputs of the attention modules are summed element-wise and the result is up-sampled through a learnable decoder.
* [Mask2Former](https://arxiv.org/abs/2112.01527): Mask2Former is a recent hybrid architecture that combines an encoder-decoder convolutional neural network with a transformer decoder to decouple the classification of classes by the segmentation, tackling different types of segmentation, e.g., semantic, instance, and panoptic segmentation. Mask2Former introduced a masking operation in the cross-attention mechanism that combines the latent vectors with the features extracted from the image, ignoring the pixel positions outside the object region.

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

**Enquiries, Question and Comments.** For enquiries, questions, or comments, please contact [Tommaso Apicella](mailto:tommaso.apicella@edu.unige.it) or open a issue in the GitHub repository.
