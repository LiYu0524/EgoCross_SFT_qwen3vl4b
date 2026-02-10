# EgoCross: Cross-Domain Egocentric Video Understanding

Fine-tuning Vision-Language Models on egocentric video understanding tasks across multiple domains.

## 📦 Resources

### Dataset
- **ModelScope**: [EgoCross_support_set](https://modelscope.cn/datasets/YuLi2024/EgoCross_support_set)

### Pre-trained Models (Full SFT, Epoch 2)
| Domain | ModelScope Link |
|--------|-----------------|
| Animal | [EgoCross_sft_qwen3vl4B_animal](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_animal) |
| Industry | [EgoCross_sft_qwen3vl4B_industry](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_industry) |
| XSports | [EgoCross_sft_qwen3vl4B_xsports](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_xsports) |
| Surgery | [EgoCross_sft_qwen3vl4B_surgery](https://modelscope.cn/models/YuLi2024/EgoCross_sft_qwen3vl4B_surgery) |

## 📊 Dataset Statistics

| Domain | Source | Samples | Frames | Description |
|--------|--------|---------|--------|-------------|
| Animal | EgoPet | 20 | 175 | Pet-mounted camera footage |
| Industry | ENIGMA | 20 | 317 | Industrial assembly operations |
| XSports | ExtremeSportFPV | 20 | 200 | First-person extreme sports |
| Surgery | CholecTrack20 | 20 | 567 | Laparoscopic surgery videos |
| **Total** | - | **80** | **1259** | - |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,qwen]"
pip install transformers>=4.57.0 qwen-vl-utils
```

### 2. Download Dataset

```bash
# Install modelscope
pip install modelscope

# Download dataset
modelscope download --dataset YuLi2024/EgoCross_support_set --local_dir ./data
```

### 3. Prepare Data

```bash
# Convert relative paths to absolute paths
python scripts/prepare_data.py --data_dir ./data --output_dir ./data_prepared
```

### 4. Train

```bash
# Full SFT (4 GPUs required)
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=4 \
    llamafactory-cli train configs/full_sft.yaml

# Or LoRA (single GPU)
llamafactory-cli train configs/lora.yaml
```

### 5. Download Pre-trained Models (Optional)

```bash
# Download a pre-trained model
modelscope download --model YuLi2024/EgoCross_sft_qwen3vl4B_surgery --local_dir ./models/surgery
```

## 📁 Repository Structure

```
egocross/
├── README.md
├── configs/
│   ├── full_sft.yaml       # Full fine-tuning config
│   └── lora.yaml           # LoRA config
├── scripts/
│   └── prepare_data.py     # Convert paths to absolute
└── data/                   # Download from ModelScope
    ├── dataset_info.json
    ├── train.json
    └── frames/
```

## ⚙️ Training Configurations

### Full SFT (Recommended)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-5 | - |
| Batch Size | 1 × 8 × 4 GPUs = 32 | gradient_accumulation=8 |
| Epochs | 1-2 | Sufficient for convergence |
| DeepSpeed | ZeRO-2 | **Required** to avoid OOM |
| image_max_pixels | 360000 | ~600×600 max |

### LoRA

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 2e-4 | Higher than Full SFT |
| LoRA Rank | 64 | - |
| LoRA Alpha | 128 | 2× rank |
| Batch Size | 2 × 4 = 8 | - |

## 📈 Results

| Method | Epochs | Surgery | Industry | XSports | Animal | Overall |
|--------|--------|---------|----------|---------|--------|---------|
| Baseline | - | 48.41 | 34.29 | 44.72 | 55.19 | 45.65 |
| Full SFT | 1 | 46.29 | 36.33 | 48.37 | 55.19 | **45.98** |
| Full SFT | 2 | 47.70 | 35.10 | 48.37 | 55.19 | **46.59** |

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM (Full SFT) | Add `deepspeed: examples/deepspeed/ds_z2_config.json` |
| CUDA OOM (images) | Set `image_max_pixels: 360000` |
| vLLM tokenizer error | Copy tokenizer files from base model |
| Dataset not found | Check `dataset_dir` path in config |

## 📝 Data Format

ShareGPT format with multi-frame video:

```json
{
  "messages": [
    {"role": "user", "content": "<image><image>...<image>Question?\nA) ...\nB) ..."},
    {"role": "assistant", "content": "A"}
  ],
  "images": ["frames/Domain/Video/frame_001.jpg", ...]
}
```

## 📜 Citation

```bibtex
@article{li2025egocross,
  title={Egocross: Benchmarking multimodal large language models for cross-domain egocentric video question answering},
  author={Li, Yanjun and Fu, Yuqian and Qian, Tianwen and Xu, Qi'ao and Dai, Silong and Paudel, Danda Pani and Van Gool, Luc and Wang, Xiaoling},
  journal={arXiv preprint arXiv:2508.10729},
  year={2025}
}
```

## 🙏 Acknowledgments

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL)
- Original datasets: EgoPet, ENIGMA, ExtremeSportFPV, CholecTrack20
