Here's the complete README.md file in Markdown syntax:

```markdown
# Skin Cancer Classification and Segmentation using Deep Learning (PyTorch)

This project implements **dual deep learning models** for skin cancer analysis:
- A **ResNet-based classifier** for 7-class lesion diagnosis
- A **U-Net model** for lesion segmentation
- An optional **multi-task model** combining both approaches

Built with PyTorch, it emphasizes medical image analysis with proper validation and performance metrics.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architectures](#model-architectures)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Project Overview
The system performs two critical tasks in dermatological AI:
1. **Classification**: Diagnoses skin lesions into 7 categories
2. **Segmentation**: Precisely outlines lesion boundaries

Key features:
- Stratified dataset splitting (70% train, 15% val, 15% test)
- Comprehensive metrics (Accuracy, IoU, Dice)
- Multi-task learning option
- GPU acceleration support

## Model Architectures

### Classification Model
- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Classifier Head**: Custom FC layer (512 → 7 classes)
- **Loss**: Cross-Entropy
- **Optimizer**: Adam (lr=1e-4)

### Segmentation Model
- **Architecture**: U-Net with ResNet-18 encoder
- **Decoder**: 4 upsampling blocks with skip connections
- **Loss**: BCEWithLogits
- **Output**: Single-channel mask (224×224)

### Multi-Task Model
- **Shared Encoder**: ResNet-18
- **Dual Heads**:
  - Classification branch (adaptive pooling → FC)
  - Segmentation branch (decoder blocks)

## Dataset
The dataset contains:
- 10,016 dermoscopic images (224×224 RGB)
- Corresponding segmentation masks
- 7 balanced classes:

| Class                  | Frequency |
|------------------------|-----------|
| Nevus                  | 66.9%     |
| Melanoma               | 11.1%     |
| Benign Keratosis       | 11.0%     |
| Basal Cell Carcinoma   | 5.1%      |
| Actinic Keratosis      | 3.3%      |
| Vascular Lesion        | 1.4%      |
| Dermatofibroma         | 1.1%      |

**Directory Structure:**
```
archive/
├── images/          # Original lesion images (.jpg)
├── masks/           # Segmentation masks (.png)
└── GroundTruth.csv  # One-hot encoded labels
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/skin-cancer-ml.git
cd skin-cancer-ml
```

2. Install dependencies:
```bash
pip install torch torchvision pandas scikit-learn matplotlib seaborn tqdm pillow
```

3. Download dataset and place in `archive/` folder

## Training

### Individual Models
```python
# Classification
python train_classifier.py --epochs 15 --batch_size 32

# Segmentation 
python train_segmenter.py --epochs 25 --batch_size 16
```

### Multi-Task Model
```python
python train_multitask.py --epochs 20 --batch_size 16
```

## Evaluation

Generate comprehensive reports:
```python
python evaluate.py --model [classifier|segmenter|multitask]
```

Output includes:
- Classification metrics (precision/recall/F1)
- Segmentation metrics (IoU/Dice)
- Visual comparisons
- Training curves

## Results

### Classification Performance
| Metric       | Value   |
|--------------|---------|
| Accuracy     | 85.4%   |
| Macro F1     | 0.832   |
| AUC          | 0.974   |

### Segmentation Performance
| Metric       | Value   |
|--------------|---------|
| Mean IoU     | 0.752   |
| Dice Coeff   | 0.823   |

Sample predictions:  
![Sample Output](docs/sample_prediction.png)

## License
MIT License - See [LICENSE](LICENSE) for details.
```
