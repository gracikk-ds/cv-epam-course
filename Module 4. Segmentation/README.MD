## Module description

###  Main things to know:
Lectures:
- [CS231n Detection and Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo&t=3433s)
- [CSC 2020: segmentation - russian](https://www.youtube.com/watch?v=NMp2i9IF9X0)

Architectures:
- [Architectures review](https://arxiv.org/pdf/2001.05566.pdf)
- (UNET and UNET++)[https://paperswithcode.com/method/u-net]
- (Mask-RCNN)[https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/README.md]
- [FPN](https://paperswithcode.com/method/fpn)

Concepts:
- What the difference between Semantic and Instance segmentation?
- What is panoptic segmentation?

Frameworks:
- Segmentation models PyTorch(https://github.com/qubvel/segmentation_models.pytorch)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) - highly recommended

###  Homework:

The current project consists of two main parts.
The first part is dedicated to the **barcodes segmentation task**, the second one to the **bars count task**

The `jobs` folder contains training jobs that you could run on Vertex AI. Check out `main.py` script from `script` folder
to learn more

The `results` folder gives you general understanding of the model's training and inference results.
