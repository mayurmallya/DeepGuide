# Deep Multimodal Guidance for Medical Image Classification

The proposed multimodal guidance strategy works as follows: (a) we first train the modality-specific classifiers C<sub>I</sub> and C<sub>S</sub> for both inferior and superior modalities, (b) next we train the guidance model G, followed by the guided inferior modality models G(I) and G(I)+I as in (c) and (d) respectively.

![](/MMG.png)


This repository contains the codes corresponding to our MICCAI 2022 paper:

Mayur Mallya, Ghassan Hamarneh, "[Deep Multimodal Guidance for Medical Image Classification](https://arxiv.org/pdf/2203.05683.pdf)", International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2022.

If you use the codes, please cite our paper. The bibtex entry is:
<!-- Add the pages in the bib -->
```
@inproceedings{mallya_2022_deepguide,
  title={Deep Multimodal Guidance for Medical Image Classification},
  author={Mallya, Mayur and Hamarneh, Ghassan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022},
  organization={Springer}
}
```
