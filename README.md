# Deep Multimodal Guidance for Medical Image Classification

Illustration of the proposed multimodal guidance strategy: (a) modality-specific classifiers C<sub>I</sub> and C<sub>S</sub>, (b) guidance model G, (c) guided model G(I), (d) guided model G(I)+I.

![](/imgs/MMG.png)


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
---

# Derm7pt

Derm7pt is a publicly available dataset of multimodal skin lesions images and can be downloaded from [here](https://github.com/jeremykawahara/derm7pt).

### Requirements

The codes used for the analysis of Derm7pt dataset are implemented mainly using Python 3.7.11, Tensorflow 2.4.1, and Cuda 10.1. The provided conda environment can be set as follows:

```
cd derm7pt
conda env create -f env_derm7pt.yml
conda activate env_derm7pt
```

### Models

#### a) Modality-specific classifiers C<sub>I</sub> and C<sub>S</sub>

#### b) Guidance model G

#### c) Guided model G(I)

#### d) Guided model G(I)+I

# RadPath

RadPath is a publicly available dataset of multimodal brain tumor images from the [RadPath 2020](https://miccai.westus2.cloudapp.azure.com/competitions/1) challenge and can be downloaded from [here](http://miccai2020-data.eastus.cloudapp.azure.com/).

### Requirements

### Models

#### a) Modality-specific classifiers C<sub>I</sub> and C<sub>S</sub>

#### b) Guidance model G

#### c) Guided model G(I)

#### d) Guided model G(I)+I
