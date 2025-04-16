# -*- encoding: utf-8 -*-
'''
    @文件名称   : augment.py
    @创建时间   : 2023/12/28 15:22:02
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 数据增强算法
    @参考地址   : 无
'''
import torch
import torchvision.transforms.functional as F

from PIL import Image
from torchvision import transforms as t



_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

class Resize:
    """

    图片缩放 (tips:  如果用来训练, 那么标签中的信息也应该相应变换)

    Attributes:
        @Param size: 缩放后的尺寸.
        @Param interpolation: 缩放策略.
        @Param resize_only: True: 仅仅缩放图片; False: 缩放图片并且修改标签信息.
    
    """
    def __init__(self, size, interpolation=Image.BILINEAR, resize_only=True):
        self.size = size
        self.interpolation = interpolation
        self.resize_only = resize_only
    
    def __call__(self, instance):
        """

        Args:
            @Param instance: 字典, 图片相关信息.
        
        Returns:
            instance: 缩放后的图片信息.

        """

        # 缩放后图片的大小.
        target_w, target_h = self.size
        # 原始图片大小.
        origin_w, origin_h = instance['image'].shape[2], instance['image'].shape[1]
 
        if (target_w-origin_w)/origin_w > (target_h-origin_h)/origin_h:
            # 高度变化相对较小
            img_h = target_h
            img_w = int(target_h/origin_h*origin_w)
            ratio = target_h/origin_h
            padding_left = (target_w-img_w)//2
            padding_right = target_w - img_w - padding_left
            padding_top = 0
            padding_bottom = 0
        else:
            # 宽度变化相对较小
            img_w = target_w
            img_h = int(target_w/origin_w*origin_h)
            ratio = target_w/origin_w
            padding_left = 0
            padding_right = 0
            padding_top = (target_h - img_h)//2
            padding_bottom = target_h - img_h - padding_top
        
        # 收集填充信息.
        padding = [padding_left, padding_top, padding_right, padding_bottom]
        # 先缩放后填充.
        instance['image'] = F.pad(F.resize(instance['image'], (img_h, img_w)), padding=padding, fill=0.0)
        
        if not self.resize_only:
            # 修改标签信息.
            for bbox in instance['bbox']:
                # 修改边界框信息
                bbox[0] = ratio * bbox[0] + padding_left
                bbox[1] = ratio * bbox[1] + padding_top
                bbox[2] = ratio * bbox[2]
                bbox[3] = ratio * bbox[3]
                # 边界框由 (l,t,w,h) 转化成 (l,t,r,b)
                bbox[2] = (bbox[0] + bbox[2])# /target_w
                bbox[3] = (bbox[1] + bbox[3])# /target_h
                bbox[2] = target_w-1 if bbox[2] >= target_w else bbox[2]
                bbox[3] = target_h-1 if bbox[3] >= target_h else bbox[3]
        instance['ratio'] = ratio
        instance['padding'] = padding
        
        return instance

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class RandomNoise:
    """
    
    随机添加噪声
    
    Attributes:
        @Param intensity: 噪声强度(0. ~ intensity).
    
    """
    def __init__(self, intensity):
        self.intensity = intensity
    
    def __call__(self, image):
        """
        Returns:
            加噪后的图片.
        
        """
        image = image + image * (torch.rand_like(image) - 0.5) * 2 * self.intensity 
        return torch.clip(image, 0.0, 1.0)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Normalize:
    """
    标准化输入
    """
    
    def __call__(self, image):
        mean = torch.mean(image, dim=(1,2)).unsqueeze(1).unsqueeze(2)
        std = torch.std(image, dim=(1,2)).unsqueeze(1).unsqueeze(2)
        return (image - mean)/std
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

def transform(instance, size, resize_only=True, add_noise=True, norm=True):
    """

    数据增强流程

    Args:
        instance: 字典, 图片信息.
        size: 缩放后的图片大小.
        resize_only: True: 仅缩放图片; False: 缩放图片并且修改标签.
        add_noise: 是否添加噪声
        norm: 是否标准化
    
    Returns:
        增强后的图片信息.
    """

    size is not None, "size cannot be None"
    # 将图片从PIL转化成张量, 图片尺寸从 (H,W,C) to (C,H,W). 
    transform = t.ToTensor()
    instance['image'] = transform(instance['image'])
    # 随机加噪.
    if add_noise:
        transform = RandomNoise(0.04)
        instance['image'] = transform(instance['image'])

    if instance['image'].shape[0] == 1:
        instance['image'] = instance['image'].repeat(3, 1, 1)
    
    # 缩放. 
    transform = Resize(size, resize_only=resize_only)
    instance = transform(instance)
    
    # 标准化
    if norm:
        transform = Normalize()
        instance['image'] = transform(instance['image'])

    return instance