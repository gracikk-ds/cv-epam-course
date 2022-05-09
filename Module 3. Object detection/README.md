## Module description

###  Main things to know:

Frameworks:
- PyTorch Detectron2
- PyTorch MMDetection
- CVAT annotation tool

Concepts:
- What is FPN ?
- What is NMS ?
- Anchor-based and Anchor-free detectors
- COCO and VOC annotations

Architectures:
- [SSD](https://arxiv.org/pdf/1512.02325.pdf)
- [YOLO](https://arxiv.org/pdf/1506.02640v5.pdf)
- [Faster RCNN](https://arxiv.org/pdf/1506.01497.pdf)
- [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf)
- [Scaled YOLOv4](https://arxiv.org/pdf/2011.08036.pdf)
- [CenterNet](https://arxiv.org/abs/1904.07850)

###  Homework:

Task:
- Take 2 or more objects (which are not presented in COCO/OpenImages)
- Take a photo of them
- Label them using CVATor VGG VIA
- Train an object detection model (using Vertex AI)

The `jobs` folder contains training jobs that you could run on Vertex AI. Check out `main.py` script from `script` folder
to learn more

The `results` folder gives you general understanding of the model's training and inference results.
