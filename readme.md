# [CVPR2022] Semi-Supervised Wide-Angle Portraits Correction by Multi-Scale Transformer
 
<h4 align="center">Fushun Zhu$^{1*}$, Shan Zhao$^{2*}$, Peng Wang$^2$, Hao Wang$^2$, Hua Yan$^1$, Shuaicheng Liu$^{3,1}$</h4>
<h4 align="center">1. Sichuan University,             2. Megvii Technology</h4>
<h4 align="center">3. University of Electronic Science and Technology of China</h4>

This is the official implementation of [**Semi-Supervised Wide-Angle Portraits Correction by Multi-Scale Transformer**](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Semi-Supervised_Wide-Angle_Portraits_Correction_by_Multi-Scale_Transformer_CVPR_2022_paper.pdf), CVPR 2022

## Abstract
We propose a semi-supervised network for wide-angle portraits correction. Wide-angle images often suffer from skew and distortion affected by perspective distortion, especially noticeable at the face regions. Previous deep learning based approaches need the ground-truth correction flow maps for training guidance. However, such labels are expensive, which can only be obtained manually. In this work, we design a semi-supervised scheme and build a high-quality unlabeled dataset with rich scenarios, allowing us to simultaneously use labeled and unlabeled data to improve performance. Specifically, our semi-supervised scheme takes advantage of the consistency mechanism, with several novel components such as direction and range consistency (DRC) and regression consistency (RC). Furthermore, different from the existing methods, we propose the Multi-Scale Swin-Unet (MS-Unet) based on the multi-scale swin transformer block (MSTB), which can simultaneously learn short-distance and long-distance information to avoid artifacts. Extensive experiments demonstrate that the proposed method is superior to the state-of-the-art methods and other representative baselines.



## Presentation Video
[[Youtube](https://www.youtube.com/watch?v=gXkn-uDcMLQ)], [[Bilibili](https://www.bilibili.com/video/BV1HU4y117ni/)]





![The pipline of semi-supervised wide-angle portraits correction framework with the surrogate task (segmentation)](https://github.com/megvii-research/Portraits_Correction/blob/main/semi-supervised%20framework.PNG)
The pipline of semi-supervised wide-angle portraits correction framework with the surrogate task (segmentation)

## Note
In this repository, we will release the unlabeled dataset and MegDL implementation of our paper.

## Quick Start

All codes are tested on Linux.

### Installation

1. Clone the repository
2. Install dependecines

### Dataset 
1.  **UltraWidePortraits2022** 
    * Unlabeled wide-angle portraits correction dataset  
    * Download from  [[GoogleDrive](https://drive.google.com/file/d/1FxzyA-EWqHnZI4H5zgZJOoqnkoAmhK0h/view?usp=sharing)] or [[BaiduCloud](https://pan.baidu.com/s/1IyeyHGR4BQHGm7Q_ZFi22g?pwd=79cw)] (extraction code:79cw)

### Pre-trained model

### Training

### Testing

## Results
![results](https://user-images.githubusercontent.com/1344482/181146762-bb2c76c1-a9d1-4786-bed9-79e8e8e740a8.JPG)


## Citation

If you think this work is helpful, please cite
```
@inproceedings{zhu2022semi,
  title={Semi-Supervised Wide-Angle Portraits Correction by Multi-Scale Transformer},
  author={Zhu, Fushun and Zhao, Shan and Wang, Peng and Wang, Hao and Yan, Hua and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19689--19698},
  year={2022}
}

```
