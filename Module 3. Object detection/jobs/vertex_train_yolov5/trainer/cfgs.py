import os
import yaml
from pathlib import Path


def dataset_yaml_file(
        root_dir,
        path_to_train_images,
        path_to_val_images,
        path_to_test_images
):
    train_list = [x.as_posix() for x in Path(path_to_train_images).glob("*")]
    valid_list = [x.as_posix() for x in Path(path_to_val_images).glob("*")]
    test_list = [x.as_posix() for x in Path(path_to_test_images).glob("*")]

    with open(os.path.join(root_dir, 'train.txt'), 'w') as f:
        for path in train_list:
            f.write(path + '\n')

    with open(os.path.join(root_dir, 'val.txt'), 'w') as f:
        for path in valid_list:
            f.write(path + '\n')

    with open(os.path.join(root_dir, 'test.txt'), 'w') as f:
        for path in test_list:
            f.write(path + '\n')

    data = dict(
        path=root_dir,
        train=os.path.join(root_dir, 'train.txt'),
        val=os.path.join(root_dir, 'val.txt'),
        test=os.path.join(root_dir, 'test.txt'),
        nc=1,
        names=['barcode'],
    )

    path_to_yaml = os.path.join(root_dir, 'gbr.yaml')

    with open(os.path.join(root_dir, 'gbr.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    return path_to_yaml


def hyp_yaml_file(root_dir):
    hyper_params = dict(
        lr0=0.001,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        lrf=0.1,  # final OneCycleLR learning rate (lr0 * lrf)
        momentum=0.9,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay 5e-4
        warmup_epochs=2.0,  # warmup epochs (fractions ok) # 3
        warmup_momentum=0.5,  # warmup initial momentum  # 0.8
        warmup_bias_lr=0.05,  # warmup initial bias lr # 0.1
        box=0.05,  # box loss gain
        cls=0.5,  # cls loss gain
        cls_pw=1.0,  # cls BCELoss positive_weight
        obj=1.0,  # obj loss gain (scale with pixels)
        obj_pw=1.0,  # obj BCELoss positive_weight
        iou_t=0.20,  # IoU training threshold
        anchor_t=3.0,  # anchor-multiple threshold
        # anchors=3,  # anchors per output layer (0 to ignore)
        fl_gamma=0.01,  # focal loss gamma (efficientDet default gamma=1.5)
        hsv_h=0.0,  # image HSV-Hue augmentation (fraction) # 0.015
        hsv_s=0.5,  # image HSV-Saturation augmentation (fraction) # .7
        hsv_v=0.4,  # image HSV-Value augmentation (fraction) # .4
        degrees=0.3,  # image rotation (+/- deg) # 0
        translate=0.1,  # image translation (+/- fraction)
        scale=0.25,  # image scale (+/- gain) # .5
        shear=0.5,  # image shear (+/- deg) # 0
        perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        flipud=0.5,  # image flip up-down (probability)
        fliplr=0.5,  # image flip left-right (probability)
        mosaic=0.1,  # image mosaic (probability)
        mixup=0,  # image mixup (probability)
        copy_paste=0.0,  # segment copy-paste (probability)
    )

    path_to_hyper_yaml = os.path.join(root_dir, 'hyp.yaml')

    with open(path_to_hyper_yaml, 'w') as outfile:
        yaml.dump(hyper_params, outfile, default_flow_style=False)

    return path_to_hyper_yaml
