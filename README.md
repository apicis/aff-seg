# Segmenting Object Affordances: Reproducibility and Sensitivity to Scale

Visual affordance segmentation identifies image regions of an object an agent can interact with. 
Existing methods re-use and adapt learning-based architectures for semantic segmentation to the affordance segmentation task and evaluate on small-size datasets. 
However, experimental setups are often not reproducible, thus leading to unfair and inconsistent comparisons. 
In this work, we benchmark these methods under a reproducible setup on two single objects scenarios, tabletop without occlusions and hand-held containers, to facilitate future comparisons. 
We include a version of a recent architecture, Mask2Former, re-trained for affordance segmentation and show that this model is the best-performing on most testing sets of both scenarios. 
Our analysis show that models are not robust to scale variations when object resolutions differ from those in the training set.


## News
* 17 August 2024: Source code, models, and further details will be released in the next weeks.
* 15 August 2024: Paper accepted at Twelfth International Workshop on Assistive Computer Vision and Robotics ([ACVR](https://iplab.dmi.unict.it/acvr2024/)), in conjunction with the 2024 European Conference on Computer Vision ([ECCV](https://eccv2024.ecva.net)).


## Contributing <a name="contributing"></a>

If you find an error, if you want to suggest a new feature or a change, you can use the issues tab to raise an issue with the appropriate label. 

Complete and full updates can be found in [CHANGELOG.md](CHANGELOG.md). The file follows the guidelines of [https://keepachangelog.com/en/1.1.0/](https://keepachangelog.com/en/1.1.0/).


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

## Enquiries, Question and Comments

If you have any further enquiries, question, or comments, or you would like to file a bug report or a feature request, please use the Github issue tracker. 


## Licence

This work is licensed under the MIT License.  To view a copy of this license, see
[LICENSE](LICENSE).