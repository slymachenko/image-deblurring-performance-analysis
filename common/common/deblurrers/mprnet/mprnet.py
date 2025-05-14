import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from collections import OrderedDict

from .Deblurring.MPRNet import MPRNet


class MPRNetDeblurrer:
    def __init__(self, weights_path: str):
        self.name = "MPRNet"
        self.img_mul_of = 8

        # Initialize the model
        self.model = MPRNet()
        self.model.cuda()

        self._load_checkpoint(self.model, weights_path)
        self.model.eval()

    def get_name(self) -> str:
        return self.name

    def deblur(self, image):
        input_ = TF.to_tensor(image).unsqueeze(0).cuda()

        # Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+self.img_mul_of)//self.img_mul_of)*self.img_mul_of, ((w+self.img_mul_of)//self.img_mul_of)*self.img_mul_of
        padh = H-h if h%self.img_mul_of!=0 else 0
        padw = W-w if w%self.img_mul_of!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            restored = self.model(input_)
            torch.cuda.empty_cache()
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)

        # Unpad the output
        restored = restored[:,:,:h,:w]

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        restored = (restored[0] * 255).round().astype(np.uint8)

        return restored

    def _load_checkpoint(self, model, weights_path):
        checkpoint = torch.load(weights_path)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
