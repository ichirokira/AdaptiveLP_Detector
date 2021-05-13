"""
Code written by TuyenNQ: s1262008@u-aizu.ac.jp
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from detectron2.structures import Boxes, ImageList, Instances

from AdaptiveLPDetection.utils import spatial_pyramid_pool, build_mask
class LP_Modules(nn.Module):
    def __init__(self, cfg):
        """
        :param pool_sizes: A List of int for sizes applying for Spartial Pyramid Pooling

        """

        super(LP_Modules, self).__init__()
        # M = cfg.LPModule.M
        # N = cfg.LPModule.N
        pool_sizes = cfg.MODEL.ADAPTIVE_LP.POOL_SIZE
        self.pool_sizes = pool_sizes
        self.kernels = []
        for i, s in enumerate(pool_sizes):
            temp = torch.empty((1,s, s), requires_grad=True) # create a learnable kernel
            nn.init.uniform_(temp, 0, 1)
            self.kernels.append(temp)
        # self.kernel1 = torch.empty((1,M, N), requires_grad=True) # create learnable kernel for LP image
        # self.kernel2 = torch.empty((1,M, N), requires_grad=True) # create learnable kernel for input image
        # nn.init.uniform_(self.kernel1, 0, 1) # intialize the kernel with uniform U(0,1)
        # nn.init.uniform_(self.kernel2, 0, 1)
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
         Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * mask: Tensor, Mask for interest region localization in (1, H, W)
                * LP_image: Tensor, local pattern image in (C, H, W) format
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        :param batched_inputs: Tuple[Dict[str, torch.Tensor]
        :return: Transformed image after fusion information from local pattern to input image in formula: W2*I + W1*LP
        """
        using_mask = self.cfg.MODEL.ADAPTIVE_LP.USING_MASK
        images, lp_images, masks = self.preprocess_image(batched_inputs)
        H, W = images.image_sizes[0]
        spp_results = []
        for i, s in enumerate(self.pool_sizes):
            padded_images, padded_mask, kernel_H, kernel_W = spatial_pyramid_pool(images=images, size=s, mask=masks)
            if using_mask is True:
                mask = build_mask(padded_mask, s, kernel_size=(kernel_H, kernel_W))
                self.kernels[i] = torch.mul(self.kernels[i], mask)
            patches = torch.nn.functional.unfold(padded_images.float(), kernel_size=(kernel_H, kernel_W), stride=(kernel_H, kernel_W))
            patches = patches.reshape((1, -1, s, s))
            patches = torch.mul(patches, self.kernels[i])
            transformed_image = torch.nn.functional.fold(patches.reshape(1, -1, s*s), output_size=(H, W),
                                              kernel_size=(kernel_H, kernel_W), stride=(kernel_H, kernel_W))
            spp_results.append(transformed_image)
        spp_results = ImageList.from_tensors(spp_results)
        result = torch.mean(spp_results.tensor, dim=0)

        return result
    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        images = [x["image"].to(self.device) for x in batched_inputs]
        lp_images = [x["LP_image"].to(self.device) for x in batched_inputs]
        masks = [x['mask'].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        lp_images = ImageList.from_tensors(lp_images)
        masks = ImageList.from_tensors(masks)
        return images, lp_images, masks