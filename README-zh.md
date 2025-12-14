# SimConfidenceEval - 仿真与真实数据置信度评估平台

一个专业的仿真-真实（Sim-Real）数据评估框架，用于衡量仿真生成数据与真实数据之间的相似度和置信度。

## 项目简介

本平台集成了**多种图像相似度评价指标**（传统指标和深度学习指标），可对成对的仿真图像和真实图像进行全面评估。支持灵活的配置系统、批处理、GPU加速和结果可视化。

## 主要特性

- 🎯 **多种评价指标**：包括 PSNR、SSIM、FSIM、LPIPS、FID 等
- ⚡ **GPU 加速**：充分利用 GPU 进行批处理加速，自动设备管理
- 📊 **灵活配置**：基于 YAML 的配置系统，支持命令行覆盖
- 📈 **结果可视化**：生成分数分布直方图、低分案例对比等
- 🔧 **模块化设计**：易于扩展新指标和数据格式
- 📝 **完整日志**：详细的运行日志和配置备份
- 🔢 **自动归一化**：支持将不同指标归一化到 [0, 1] 范围，便于统一比较

## 项目结构

```
SimConfidenceEval/
├── main.py                          # 主入口脚本
├── requirements.txt                 # 依赖包列表
├── configs/                         # 配置文件目录
│   ├── basic_eval.yaml             # 基础评估配置
│   └── dataset_eval.yaml           # 数据集评估配置
├── data/                            # 数据文件目录
│   ├── paired/                         # 配对评估用（局部/成对指标）
│   │   ├── real/
│   │   │   └── <category>/
│   │   │       ├── xxx.jpg
│   │   │       ├── xxx.txt             # 可选：YOLO 标注（与图像同名）
│   │   │       └── ...
│   │   └── sim/
│   │       └── <category>/
│   │           ├── xxx.jpg
│   │           ├── xxx.txt             # 可选：YOLO 标注（与图像同名）
│   │           └── ...
│   └── global/                         # 全局评估用（分布指标，如 FID）
│       ├── real/
│       │   └── ...                     # 真实数据全集
│       └── sim/
│           └── ...                     # 仿真数据全集
├── src/                             # 源代码
│   ├── core/                       # 核心模块
│   │   └── evaluator.py           # 评估器主类
│   ├── data/                       # 数据处理模块
│   │   ├── datasets.py            # 数据集定义和加载
│   │   ├── transforms.py          # 数据增强和转换
│   │   └── utils.py               # 数据处理工具
│   ├── metrics/                    # 评价指标模块
│   │   ├── base.py                # 基类定义
│   │   ├── classical.py           # 古典指标（PSNR/SSIM/FSIM）
│   │   └── deep.py                # 深度学习指标（LPIPS/FID）
│   └── utils/                      # 工具模块
│       ├── logger.py              # 日志工具
│       ├── io.py                  # 文件 I/O 工具
│       └── visualizer.py          # 可视化工具
├── test/                            # 测试代码
│   └── test_metrics.py            # 指标单元测试
└── outputs/                         # 输出结果目录
    └── run_001/                    # 示例运行结果
```

## 支持的评价指标

| 指标 | 类型 | 说明 |
|------|------|------|
| **PSNR** | 传统 | 峰值信噪比，越高越相似 |
| **SSIM** | 传统 | 结构相似度，考虑亮度、对比度、结构 |
| **FSIM** | 传统 | 特征相似度，基于相位一致性 |
| **LPIPS** | 深度学习 | 感知损失，基于预训练神经网络提取特征 |
| **FID** | 深度学习 | Fréchet 初始距离，评估整体分布相似度 |

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用 conda
conda env create -f environment.yml
```

### 2. 准备数据

将仿真和真实数据按如下结构放置：

```
data/
├── sim/
│   └── <category>/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── real/
    └── <category>/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

### 3. 配置评估参数

编辑 `configs/basic_eval.yaml`：

```yaml
experiment_name: "my_experiment"
output_dir: "./outputs/run_001"

data:
  mode: "paired"                    # 配对模式
  sim_path: "./data/sim/"
  real_path: "./data/real/"
  iou_threshold: 0.8               # IoU 过滤阈值
  batch_size: 32
  image_size: [640, 480]           # [H, W]

metrics:
  - name: "PSNR"
    params:
      psnr_max: 50.0
  - name: "SSIM"
  - name: "LPIPS"
    params:
      net: "vgg"
      lpips_max: 1.0
  - name: "FID"
    params:
      feature: 2048

visualization:
  plot_hist: True                  # 绘制直方图
  save_bad_cases: True             # 保存低分案例
```

### 4. 运行评估

```bash
# 使用默认配置
python main.py

# 指定配置文件
python main.py --config configs/basic_eval.yaml

# 覆盖输出目录
python main.py --config configs/basic_eval.yaml --output ./outputs/custom_run
```

### 5. 计算 FID 上下限（归一化基线）

使用 `calculate_global_best_worst.py` 计算参考集的 FID 最佳/最差基线，并保存到 `fid_baselines.txt`：

```powershell
python .\calculate_global_best_worst.py
```

说明：
- FID_best：将参考集随机切分为两半，多次计算 FID，取中位数。
- FID_worst：将参考集与独立图像集（CIFAR-10 或 ImageNet）计算 FID，作为下限。
- 归一化公式：$S_{fid} = \frac{FID_{worst} - FID}{FID_{worst} - FID_{best}}$，范围 [0, 1]。

## 输出结果

运行完成后，输出目录包含：

```
outputs/run_001/
├── config_backup.yaml                    # 配置文件备份
├── final_report_summary.csv              # 摘要报告（每个类别的均值和标准差）
├── final_report_detailed.csv             # 详细报告（每对图像的详细得分）
├── plots/
│   ├── PSNR_dist.png                    # 分数分布直方图
│   ├── SSIM_dist.png
│   └── ...
└── logs/
    └── eval_*.log                       # 运行日志
```

### 输出文件说明

**详细报告 (final_report_detailed.csv)**：
- 按类别分组，每个类别包含：
  - 类别标题行
  - 每对图像的原始指标得分
  - 空列分隔符
  - 每对图像的归一化指标得分
  - MEAN 行：各指标的均值（原始值和归一化值）
  - STD 行：各指标的标准差（原始值和归一化值）

**简略报告 (final_report_summary.csv)**：
- 每个类别一行
- 列结构：category | {metric}_mean | {metric}_std | 空列 | {metric}_norm_mean | {metric}_norm_std
- 便于快速对比不同类别的整体表现

## 配置详解

### data 配置

- `mode`：数据加载模式
  - `paired`：配对模式，要求 sim 和 real 目录结构完全相同
- `iou_threshold`：框过滤阈值，用于检测框级别的质量控制
- `batch_size`：批处理大小，调整以适应 GPU 显存
- `image_size`：统一图像尺寸 `[高, 宽]`

### metrics 配置

每个指标可包含 `name` 和可选的 `params`：

- **PSNR** 参数：
  - `psnr_max`：用于归一化的上限值（默认 50）

- **LPIPS** 参数：
  - `net`：特征网络 `alex` / `vgg` / `squeeze`
  - `version`：模型版本
  - `lpips_max`：归一化上限（默认 1.0）

- **FID** 参数：
  - `feature`：特征维度 `64` / `192` / `768` / `2048`

### visualization 配置

- `plot_hist`：是否生成分数分布直方图
- `save_bad_cases`：是否保存得分最低的对比图像

## 扩展指南

### 添加新的评价指标

1. **在 `src/metrics/` 中创建指标类**

```python
from src.metrics.base import BaseMetric
import torch

class MyMetric(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # 初始化你的指标
        
    def forward(self, preds, target):
        # 实现计算逻辑
        score = ...  # 计算相似度
        return self._normalize(score)
    
    def _normalize(self, x):
        # 归一化到 [0, 1]
        return x / max_value
```

2. **在 `src/metrics/__init__.py` 中注册**

```python
from .classical import PSNR, SSIM, FSIM
from .deep import LPIPS, FID
from .custom import MyMetric  # 导入新指标
```

3. **在配置文件中使用**

```yaml
metrics:
  - name: "MyMetric"
    params:
      param1: value1
```

### 支持新数据格式

修改 `src/data/datasets.py` 中的 `build_dataloader` 函数以支持其他数据格式（如检测框标注、点云等）。

## 性能建议

- **使用 GPU**：确保 CUDA 可用，评估速度可提升 10-100 倍
  - 框架自动检测并管理 GPU/CPU 设备切换
  - FID 等全局指标已优化，确保在 GPU 上高效运行
- **批大小调整**：根据 GPU 显存调整 `batch_size`（推荐 32-128）
- **图像尺寸**：较小尺寸加速处理，较大尺寸提升精度（推荐 640x480）
- **指标选择**：LPIPS/FID 较慢，优先使用古典指标进行快速评估
- **归一化结果**：在配置中启用 `need_normalize_result` 可同时获得原始和归一化分数

## 依赖说明

主要依赖包括：

- PyTorch：深度学习框架
- PyYAML：配置文件解析
- scikit-image：图像处理算法
- Pillow：图像 I/O
- matplotlib：可视化
- pandas：数据处理

详见 `requirements.txt`

## 常见问题

**Q: 如何处理尺寸不一致的图像？**  
A: 框架会自动 resize 所有图像到配置中的 `image_size`。

**Q: 支持哪些图像格式？**  
A: 支持 PIL 支持的所有格式（JPEG, PNG, BMP 等）。

**Q: 如何在 CPU 上运行？**  
A: 框架会自动检测 CUDA 可用性，无可用 GPU 时自动使用 CPU。

**Q: 如何跳过某些指标的计算？**  
A: 在配置文件中删除或注释掉不需要的指标即可。

## 许可证

MIT

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
