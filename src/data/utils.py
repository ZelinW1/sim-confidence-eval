import os
import numpy as np
from pathlib import Path

def find_matching_file(base_dir, relative_path, target_extensions=['.jpg', '.png', '.jpeg', '.bmp']):
    """
    在 base_dir 下寻找与 relative_path (不含后缀) 匹配的图像文件，支持多种后缀。
    :param base_dir: 根目录 (e.g. ./data/sim)
    :param relative_path: 相对路径不含后缀 (e.g. ship/1_123)
    :return: 完整路径 or None
    """
    base_path = Path(base_dir) / relative_path
    # 先尝试直接拼接每一个后缀
    for ext in target_extensions:
        candidate = base_path.with_suffix(ext)
        if candidate.exists():
            return str(candidate)
    return None

def parse_yolo_box(txt_path, img_width, img_height):
    """
    解析 YOLO 格式标注文件，返回 (x1, y1, x2, y2) 绝对坐标。
    假设文件只有一行或者只取第一行。
    YOLO format: class_id x_center y_center w h (归一化)
    """
    if not os.path.exists(txt_path):
        return None

    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return None

    # 取第一行有效数据
    parts = lines[0].strip().split()
    if len(parts) < 5:
        return None

    # 解析
    _, x_c, y_c, w, h = map(float, parts[:5])

    # 转换为绝对坐标
    x_c *= img_width
    y_c *= img_height
    w *= img_width
    h *= img_height

    x1 = int(max(0, x_c - w / 2))
    y1 = int(max(0, y_c - h / 2))
    x2 = int(min(img_width, x_c + w / 2))
    y2 = int(min(img_height, y_c + h / 2))

    return (x1, y1, x2, y2)

def compute_iou(box1, box2):
    """
    计算两个矩形框的 IoU
    box: (x1, y1, x2, y2)
    """
    if box1 is None or box2 is None:
        return 0.0

    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # 计算交集
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # 计算并集
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area

    if union_area <= 1e-6:
        return 0.0

    return inter_area / union_area