# Check Pytorch installation
import torch, torchvision
print("torch version:",torch.__version__, "cuda:",torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print("mmdetection:",mmdet.__version__)

# Check mmcv installation
import mmcv
print("mmcv:",mmcv.__version__)

# Check mmengine installation
import mmengine
print("mmengine:",mmengine.__version__)

import numpy as np
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
import matplotlib.pyplot as plt

from mmdet.registry import VISUALIZERS
import logging

logging.getLogger().setLevel(logging.WARNING)

def inference(path_to_image):
    
    # Choose to use a config and initialize the detector
    config_file = '/home/nfierroflo/Documents/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint_file = '/home/nfierroflo/Documents/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    image = mmcv.imread(path_to_image,channel_order='rgb')
    result_left = inference_detector(model, image)

    #plt.imshow(result_left._pred_instances.masks[0])
    persons=result_left._pred_instances.masks[0]
    X=np.array([])
    Y=np.array([])
    for iy, ix in np.ndindex(persons.shape):
                if (persons[iy, ix]==True):
                    X=np.append(X,ix)
                    Y=np.append(Y,iy)

    return X,Y,result_left._pred_instances.masks[0]

def new_inference(path_to_image):

    # Choose to use a config and initialize the detector
    config_file = '/home/nfierroflo/Documents/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint_file = '/home/nfierroflo/Documents/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    image = mmcv.imread(path_to_image,channel_order='rgb')
    result_left = inference_detector(model, image)


    results= (result_left._pred_instances.masks.cpu(),result_left._pred_instances.scores.cpu(),result_left._pred_instances.labels.cpu())
    return results

def generate_mask(path_to_image):
    # Choose to use a config and initialize the detector
    config_file = '/home/nfierroflo/Documents/mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint_file = '/home/nfierroflo/Documents/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

    image = mmcv.imread(path_to_image,channel_order='rgb')
    result_left = inference_detector(model, image)

    return result_left
       
