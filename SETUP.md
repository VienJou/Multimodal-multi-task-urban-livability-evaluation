# Setup and Running Instructions

This guide explains how to set up and run the notebooks on a new server.

## 📋 Prerequisites

1. **Python Environment**: Python 3.8+ with required packages (see `requirements.txt`)
2. **Git**: To clone the repository
3. **GPU** (recommended): CUDA-capable GPU for faster training
   - **CPU is supported**: The code will automatically detect and use CPU if no GPU is available
   - **Note**: Training on CPU will be significantly slower (10-100x slower than GPU)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone git@github.com:VienJou/Multimodal-multi-task-urban-livability-evaluation.git
cd Multimodal-multi-task-urban-livability-evaluation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Working Directory

**Important**: The notebooks use **relative paths** and will automatically detect the project directory. Make sure to:

1. Navigate to the project directory:
```bash
cd Multimodal-multi-task-urban-livability-evaluation
```

2. Start Jupyter from this directory:
```bash
jupyter notebook
```

Or if using JupyterLab:
```bash
jupyter lab
```

### 4. Run the Notebooks

The notebooks will automatically:
- Detect the project directory based on the current working directory
- Find required modules (`textBert_utils.py`, `MMBT_liva/`, etc.)
- Resolve paths relative to the project directory

#### Main Training/Evaluation Notebook
```bash
# Open in Jupyter
jupyter notebook Livability_evaluation_baseline_EN_Clean_v5.ipynb
```

#### Dataset Showcase Notebook
```bash
# Open in Jupyter
jupyter notebook Livability_Dataset_Showcase.ipynb
```

## 📁 Project Structure

The code expects the following structure (all relative paths):

```
Multimodal-multi-task-urban-livability-evaluation/
├── Livability_evaluation_baseline_EN_Clean_v5.ipynb  # Main notebook
├── Livability_Dataset_Showcase.ipynb                 # Dataset showcase
├── textBert_utils.py                                 # Utility functions
├── MMBT_liva/                                        # Model modules
│   ├── image_liva.py
│   ├── mmbt_config_liva.py
│   ├── mmbt_liva.py
│   └── mmbt_utils_liva_0318.py
├── livability_4M_6aspects/                           # Model outputs (created during training)
│   ├── config.json
│   ├── tokenizer*.json
│   └── vocab.txt
└── requirements.txt
```

## 🔧 Path Resolution

The notebooks use **automatic path resolution**:

1. **Module Import Path** (Cell 5):
   - Automatically searches for `textBert_utils.py` in multiple locations
   - Adds the found directory to `sys.path`
   - Works regardless of where you start Jupyter from (as long as you're in or above the project directory)

2. **Output Directory Path** (Cell 22 & 25):
   - Automatically finds the `Livability_evaluation_baseline` directory
   - Resolves `output_dir` relative to the project directory
   - Default: `livability_4M_6aspects` (relative path)

## 📊 Dataset

The dataset is loaded from Hugging Face automatically:
- **Dataset**: `Vinjou/Multimodal_urban_livability_evaluation_dataset`
- No local dataset files needed
- First run will download the dataset (cached for subsequent runs)

### 🔐 Hugging Face Authentication

**Important**: The dataset requires Hugging Face authentication. You need to login before running the notebook:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login
```

When prompted, enter your Hugging Face token. You can get your token from:
- https://huggingface.co/settings/tokens

**Alternative methods** (if CLI login doesn't work):

1. **Set environment variable**:
```bash
export HF_TOKEN=your_token_here
```

2. **In Python/Jupyter** (before loading dataset):
```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'
```

3. **Direct token in code** (modify `load_examples` function):
```python
hf_dataset_dict = load_dataset("Vinjou/Multimodal_urban_livability_evaluation_dataset", token="your_token_here")
```

The code will automatically detect and use your token if you've logged in via `huggingface-cli login`.

## ⚙️ Configuration

### Default Settings

- **Output Directory**: `livability_4M_6aspects` (relative to project directory)
- **Model**: BERT-base-multilingual-uncased
- **Image Encoder**: DenseNet-121 (ImageNet pre-trained, auto-downloaded)

### Customizing Paths

If you need to customize paths, modify the arguments in **Cell 14**:

```python
parser.add_argument(
    "--output_dir",
    default="livability_4M_6aspects",  # Relative path
    type=str,
    help="The output directory where the model predictions and checkpoints will be written.",
)
```

## 🐛 Troubleshooting

### Issue: Module not found

**Solution**: Make sure you're running Jupyter from the project directory or a parent directory. The code will automatically search for modules.

### Issue: Path not found

**Solution**: 
1. Check that you're in the correct directory
2. Verify the project structure matches the expected layout
3. The code will print debug information showing which paths it's checking

### Issue: Dataset download fails / ConnectionError / Unauthorized

**Solution**:
1. **Authentication required**: Run `huggingface-cli login` and enter your token
2. Check your internet connection
3. Verify you have access to the dataset (check dataset page on Hugging Face)
4. Check disk space for dataset cache
5. If using a web platform, ensure the platform allows Hugging Face access

**Common Error**: `ConnectionError: Unauthorized for URL... Please use the parameter token=True`
- This means you need to authenticate. Run `huggingface-cli login` first.

### Issue: Running on CPU (No GPU)

**Solution**: The code automatically detects and uses CPU if no GPU is available. However, for better performance on CPU:

1. **Reduce batch size** in Cell 14 (Argument Parsing):
   ```python
   parser.add_argument("--train_batch_size", default=4, type=int, ...)  # Reduce from 16 to 4 or 2
   parser.add_argument("--eval_batch_size", default=4, type=int, ...)    # Reduce from 16 to 4 or 2
   ```

2. **Reduce number of workers**:
   ```python
   parser.add_argument("--num_workers", type=int, default=2, ...)  # Reduce from 8 to 2 or 0
   ```

3. **Use smaller dataset subset** for testing (modify `load_examples` to use a subset)

4. **Expected performance**:
   - GPU: Training typically takes hours
   - CPU: Training may take days or weeks depending on dataset size
   - Evaluation on CPU is more feasible but still slower

**Note**: The code will automatically print "No GPU available, using the CPU instead." when running on CPU.

## 📝 Notes

- All paths in the code are **relative paths**
- The code automatically adapts to different directory structures
- No absolute paths are hardcoded (except in comments, which are informational only)
- Model checkpoints and outputs are saved relative to the project directory

## 🔗 Additional Resources

- Dataset: [Hugging Face Dataset](https://huggingface.co/datasets/Vinjou/Multimodal_urban_livability_evaluation_dataset)
- Paper: [Link to paper](https://www.sciencedirect.com/science/article/pii/S0034425726000027)
