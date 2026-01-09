# Enhancing Intracranial Vessel Segmentation using Diffusion Models without Label for 3D Time-of-Flight Magnetic Resonance Angiography

[![arXiv](https://img.shields.io/badge/CMIG-102651-222A6C.svg?style=plastic)](https://www.sciencedirect.com/science/article/pii/S0895611125001600)

## Overview

This repository contains the code for Enhancing Intracranial Vessel Segmentation using Diffusion Models without Label for 3D Time-of-Flight Magnetic Resonance Angiography. The pipeline to find the threshold value and then get the threshold label is shown in the figure below:

![fig2](./assets/threshold.png)



Afterward, the diffusion model below can be trained and inferred using the threshold label.



![model](./assets/model.png)



## Citation

```tex
@article{kim2025enhancing,
  title={Enhancing intracranial vessel segmentation using diffusion models without manual annotation for 3D Time-of-Flight Magnetic Resonance Angiography},
  author={Kim, Jonghun and Na, Inye and Chung, Jiwon and Song, Ha-Na and Kim, Kyungseo and Ju, Seongvin and Eun, Mi-Yeon and Seo, Woo-Keun and Park, Hyunjin},
  journal={Computerized Medical Imaging and Graphics},
  pages={102651},
  year={2025},
  publisher={Elsevier}
}
```

