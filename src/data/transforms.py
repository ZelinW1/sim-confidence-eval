import torch
from PIL import Image
from torchvision import transforms

class ImagePreprocessor:
    def __init__(self, cfg):
        self.img_size = cfg.get("image_size", [256, 256]) # [H, W]
        
        # 基础的转 Tensor 和 归一化
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # 默认归一化到 [0, 1]，如果模型需要 Standardize 可以在这里加
        ])

    def load_and_process(self, img_path, box=None):
        """
        读取图片 -> (可选) ROI裁剪 -> Resize -> ToTensor
        :param box: (x1, y1, x2, y2) 如果不为None，则先裁剪
        """
        # 1. Load Image (RGB)
        img = Image.open(img_path).convert('RGB')
        
        # 2. Crop ROI if box provided
        if box is not None:
            # PIL crop args: (left, upper, right, lower)
            img = img.crop(box)
        
        # 3. Resize
        # Image.LANCZOS is high quality downsampling filter
        img = img.resize((self.img_size[1], self.img_size[0]), Image.Resampling.LANCZOS)
        
        # 4. To Tensor
        tensor = self.to_tensor(img)
        return tensor

    def get_image_size(self, img_path):
        """仅获取图片尺寸而不读取全部数据，用于快速解析 YOLO"""
        with Image.open(img_path) as img:
            return img.size # (width, height)