# Multimodal Multi-task Urban Livability Evaluation

This repository contains the implementation of a multimodal (text and image) regression model for urban livability evaluation using the MMBT (Multimodal Bitransformer) architecture.

## 📖 Paper

This project implements the methodology described in:

**Paper Title**: [A transformer based multi-task deep learning model for urban livability evaluation by fusing remote sensing and textual geospatial data](https://www.sciencedirect.com/science/article/pii/S0034425726000027)
If you use this code and dataset in your research, please cite:

```bibtex
@article{zhou2026,
  title={A transformer based multi-task deep learning model for urban livability evaluation by fusing remote sensing and textual geospatial data},
  author={Zhou, Wen and Persello, Claudio and Ming, Dongping and Wang, Shaowen and Stein, Alfred},
  journal={Remote Sensing of Environment},
  volume={334},
  pages={115232},
  year={2026},
  doi={10.1016/j.rse.2026.115232}
}
```

## 🎯 Overview

This project fine-tunes a pre-trained MMBT (Multimodal Bitransformer) model for multi-task regression of urban livability across 6 aspects:
- **Livability**: Overall livability -- (Dutch: LBM)
- **PHY**: Physical Environment -- (Dutch: FYS)
- **NUI**: Nuisance and Insecurity -- (Dutch: ONV)
- **SOC**: Social cohesion -- (Dutch: SOC)
- **AME**: Amenities -- (Dutch: VRZ)
- **HOU**: Housing quality -- (Dutch: WON)

The model combines:
- **Text features**:  Point of interes data
- **Image features**: Three-channel satellite imagery (RS, DSM, NLRS(GIU)) using DenseNet-121 encoder

## 📊 Dataset

The dataset is publicly available on Hugging Face:

**Dataset**: [Vinjou/Multimodal_urban_livability_evaluation_dataset](https://huggingface.co/datasets/Vinjou/Multimodal_urban_livability_evaluation_dataset)

You can load the dataset directly using Hugging Face datasets:

```python
from datasets import load_dataset

dataset = load_dataset("Vinjou/Multimodal_urban_livability_evaluation_dataset")
```

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.0+
- CUDA-capable GPU (recommended)

### Setup

1. Clone this repository:
```bash
git clone git@github.com:VienJou/Multimodal-multi-task-urban-livability-evaluation.git
cd Multimodal-multi-task-urban-livability-evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision transformers datasets
pip install numpy pandas matplotlib seaborn
pip install tensorboard
```

3. Download the pre-trained image encoder model:
   - The code expects `saved_chexnet.pt` (DenseNet-121 pre-trained on CheXNet) in `data_livability/models/`
   - You may need to download this separately or train it yourself

## 📁 Project Structure

```
Livability_evaluation_baseline/
├── README.md                                    # This file
├── Livability_evaluation_baseline_EN_Clean_v5.ipynb  # Main training/evaluation notebook
├── textBert_utils.py                           # Utility functions for BERT/text processing
├── MMBT_liva/
│   ├── mmbt_config_liva.py                     # MMBT configuration class
│   ├── mmbt_liva.py                            # MMBT model implementation
│   ├── image_liva.py                           # Image encoder (DenseNet-121)
│   └── mmbt_utils_liva_0318.py                 # Dataset loading and utilities
├── livability_4M_6aspects/                      # Model checkpoints and outputs
│   ├── pytorch_model.bin                       # Final trained model
│   ├── checkpoint-*/                           # Training checkpoints
│   └── eval_results.txt                        # Evaluation results
└── data_livability/                            # Data directory (not included in repo)
    └── models/
        └── saved_chexnet.pt                     # Pre-trained image encoder
```

## 💻 Usage

### Training

1. Open the Jupyter notebook:
```bash
jupyter notebook Livability_evaluation_baseline_EN_Clean_v5.ipynb
```

2. Configure the training parameters in **Cell 14** (Argument Parsing):
   - `--output_dir`: Output directory for model checkpoints (default: `livability_4M_6aspects`)
   - `--train_batch_size`: Batch size for training (default: 16)
   - `--num_train_epochs`: Number of training epochs (default: 1)
   - `--learning_rate`: Learning rate (default: 5e-5)
   - And other hyperparameters...

3. Run the cells sequentially:
   - **Cell 1-4**: Setup and imports
   - **Cell 5**: Path configuration
   - **Cell 6-13**: Model and dataset setup
   - **Cell 14**: Argument parsing
   - **Cell 15-21**: Model initialization
   - **Cell 22**: Training setup
   - **Cell 23**: Training loop
   - **Cell 24**: Final model saving
   - **Cell 25**: Evaluation

### Evaluation

The evaluation code automatically:
- Loads the trained model from `output_dir`
- Evaluates on test set
- Generates scatter plots and evaluation metrics
- Saves results to `eval_results.txt`

To run evaluation only, set:
```python
args.do_train = False
args.do_eval = True
```

### Using Pre-trained Model

If you want to use the pre-trained model for inference:

```python
from transformers import AutoTokenizer, AutoModel
from MMBT_liva.mmbt_liva import MMBTForClassification  # Note: class name is Classification but used for regression
from MMBT_liva.image_liva import ImageEncoderDenseNet
from MMBT_liva.mmbt_config_liva import MMBTConfig

# Load model
model_path = "livability_4M_6aspects"
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
# ... (see notebook for full example)
```

## 🔧 Configuration

### Model Architecture

- **Text Encoder**: BERT-base-multilingual-uncased (12 layers, 768 hidden size)
- **Image Encoder**: DenseNet-121 (pre-trained on CheXNet)
- **Image Input**: 3 images concatenated (RS + DSM + GIU = 9 channels)
- **Number of Labels**: 6 (multi-task regression)
- **Modal Hidden Size**: 1024

### Training Hyperparameters

- **Batch Size**: 16
- **Learning Rate**: 5e-5
- **Max Sequence Length**: 400
- **Number of Image Embeddings**: 9
- **Optimizer**: AdamW
- **Weight Decay**: 0.1

## 📈 Results

Evaluation results are saved in:
- `livability_4M_6aspects/eval_results.txt`: Detailed evaluation metrics
- `livability_4M_6aspects/final_eval_results.txt`: Final evaluation summary

The model outputs predictions for all 6 livability aspects with correlation metrics and scatter plots.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- This implementation is adapted from the Hugging Face [MMBT implementation](https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/run_mmimdb.py)
- The MMBT architecture is based on the work by Kiela et al. (2020)
- Image encoder uses DenseNet-121 pre-trained on CheXNet

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].

---

**Note**: This repository is part of ongoing research. Please refer to the paper for detailed methodology and results.

