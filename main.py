import argparse
import yaml
import os
import sys
from pathlib import Path

# 确保 src 目录在系统路径中，防止 import 报错
sys.path.append(str(Path(__file__).parent))

from src.core import Evaluator
from src.utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Sim-Real Confidence Evaluation Platform")
    
    # 核心参数：配置文件路径
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/basic_eval.yaml", 
        help="Path to the yaml configuration file."
    )
    
    # 可选参数：覆盖输出目录（方便调试）
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Override output directory specified in config."
    )
    
    return parser.parse_args()

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)
    return cfg

def main():
    # 1. 解析参数
    args = parse_args()
    
    # 2. 加载配置
    print(f"Loading configuration from {args.config}...")
    cfg = load_config(args.config)
    
    # 如果命令行指定了 output，覆盖 yaml 中的设置
    if args.output:
        cfg['output_dir'] = args.output
        
    # 确保输出目录存在
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    # 3. 备份配置文件到输出目录（方便实验溯源）
    dst_config_path = os.path.join(cfg['output_dir'], 'config_backup.yaml')
    with open(dst_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # 4. 初始化并运行评估器
    try:
        evaluator = Evaluator(cfg)
        evaluator.run()
        print(f"\n✅ Evaluation completed successfully! Check results in: {cfg['output_dir']}")
    except Exception as e:
        # 捕获运行时的未预料错误并记录
        print(f"\n❌ An error occurred during evaluation:")
        print(str(e))
        # 打印详细 traceback 用于调试
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()