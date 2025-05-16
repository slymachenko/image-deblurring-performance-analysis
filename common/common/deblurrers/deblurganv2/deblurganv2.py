import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as albu
from .fpn_inception import FPNInception
import functools

class DeblurGANv2Deblurrer:
    def __init__(self, weights_path: str):
        self.name = "deblurganv2"

        # Define normalization layer for instance normalization
        def get_norm_layer(norm_type='instance'):
            if norm_type == 'instance':
                return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
            else:
                raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        # Initialize the model
        self.model = nn.DataParallel(FPNInception(norm_layer=get_norm_layer(norm_type='instance'))).cuda()
        self.model.load_state_dict(torch.load(weights_path)['model'])
        self.model.train(True)  # Keep in train mode for norm layer stats

        # Define normalization function
        self.normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_name(self) -> str:
        return self.name

    def deblur(self, img) -> None:
        h, w, _ = img.shape

        # Preprocess: Normalize and pad the image
        img = self.normalize(image=img)['image']
        block_size = 32
        min_height = (h // block_size + 1) * block_size if h % block_size != 0 else h
        min_width = (w // block_size + 1) * block_size if w % block_size != 0 else w
        img_padded = np.pad(img, ((0, min_height - h), (0, min_width - w), (0, 0)), 
                           mode='constant', constant_values=0)

        # Convert to tensor
        x = np.transpose(img_padded, (2, 0, 1))  # HWC to CHW
        x = np.expand_dims(x, 0)  # Add batch dimension
        x = torch.from_numpy(x).float().cuda()

        # Perform prediction
        with torch.no_grad():
            pred = self.model(x)
            torch.cuda.empty_cache()

        # Postprocess: Convert back to image format and remove padding
        pred = pred[0].detach().cpu().numpy()
        pred = np.transpose(pred, (1, 2, 0))  # CHW to HWC
        pred = (pred + 1) / 2.0 * 255.0  # Rescale from [-1,1] to [0,255]
        pred = pred.astype('uint8')
        pred = pred[:h, :w, :]  # Remove padding

        return pred