# Swin Visformer for Object Detection

This repo contains the code of object detection for [Visformer](?). It is based on [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and [mmdetection](https://github.com/open-mmlab/mmdetection).


## Object Detection on COCO
The standard self-attention is not efficient for high-reolution inputs, 
so we simply replace the standard self-attention with Swin attention for object detection. Therefore, Swin Transformer is our directly baseline. 
### Mask R-CNN
| Backbone | sched | box mAP | mask mAP | params | FLOPs | FPS |
| :---: | :---: |  :---: | :---: |  :---: |  :---: | :---: | 
| Swin-T |1x| 42.6 | 39.3 | 48 | 267 | 14.8 |
| Visformer-S | 1x| 43.0 | 39.6 | 60 | 275 | 13.1|
| VisformerV2-S | 1x| 44.8 | 40.7 | 43 | 262 | 15.2 |
|Swin-T |3x + MS|  46.0 | 41.6 | 48 | 367 | 14.8 |
| VisformerV2-S | 3x + MS| 47.8 | 42.5 | 43 | 262 | 15.2 |

### Cascade Mask R-CNN
| Backbone | sched | box mAP | mask mAP | params | FLOPs | FPS |
| :---: | :---: |  :---: | :---: |  :---: |  :---: | :---: |
| Swin-T |1x + MS|  48.1 | 41.7 | 86 | 745 | 9.5 |
| VisformerV2-S |1x + MS|  49.3 | 42.3 | 81 | 740 | 9.6 |
| Swin-T |3x + MS|  50.5 | 43.7 | 86 | 745 | 9.5 |
| VisformerV2-S |3x + MS|  51.6 | 44.1 | 81 | 740 | 9.6 |

## Usage
(Inherited from Swin Transformer)

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation and dataset preparation.
(mmcv == 1.3.9)
### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a VisformerV2_S backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin_visformer/cascade_mask_rcnn_swin_visformer_small_v2_mstrain_480-800_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Other Links

> **Visformer for Classification**: See [Visformer for Image Classification](https://github.com/danczs/Visformer).
