import logging
import os
import sys

def setup_logger(output_dir, name="eval"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 2. 文件输出
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, f'{name}.log'), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger