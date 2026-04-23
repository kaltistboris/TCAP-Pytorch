# TCAP-PyTorch

这是一个基于 PyTorch 和 `scikit-survival` (sksurv) 重构的 **TCAP** (Transfer-learning based Cox Proportional Hazards Network) 实现。

## 📌 项目背景与改进
本项目是对原论文算法 [TCAP (fsct135/TCAP)](https://github.com/fsct135/TCAP) 的现代化重构。

**主要改进点：**
- **框架迁移**：从原生的 NumPy/自定义/TensorFlow 实现迁移到了 **PyTorch**，支持 GPU 加速和自动求导。
- **生存分析集成**：使用 **scikit-survival** 处理生存数据模型和评估指标（如 C-Index）。
- **环境管理**：使用 **uv** 进行包管理，确保环境一键配置。
- **模块化设计**：重构了数据加载和网络结构，代码更易于扩展和阅读。

## 🛠 快速开始

### 1. 环境准备
本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行管理。

```bash
# 安装依赖并创建虚拟环境
uv sync
```

### 2. 运行项目
```bash
# 使用 uv 运行主程序
uv run main.py
```

## 📂 文件结构说明
- `main.py`: 具体的使用示例。
- `dsl.py`: 无迁移学习功能的模型代码。
- `utils.py`: 数据预处理和辅助函数。
- `incremental.py`: 支持迁移学习相关的代码。
- `pyproject.toml`: 项目依赖定义。

### 使用说明
- `main.py` 中包含了一个示例流程，展示了如何加载数据、训练模型和评估性能。
- 你可以根据需要修改数据加载部分，适配自己的生存分析数据集。
- 此外如果不需要迁移学习功能，可以直接使用 `dsl.py` 中的模型实现，大致流程和 `main.py` 类似。

## 📜 致谢
本项目基于以下研究工作进行开发：
- 原作者仓库: [fsct135/TCAP](https://github.com/fsct135/TCAP)
- 原始描述: A transfer-learning based Cox proportional hazards network (TCAP).