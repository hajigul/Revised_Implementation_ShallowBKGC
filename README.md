# Revised_Implementation_ShallowBKGC

This repository contains an enhanced implementation of ShallowBKGC (Shallow Knowledge Graph Completion) that combines:

##  Performance Metrics

The model evaluates relation prediction using standard KG completion metrics:
- **MR (Mean Rank)**: Average rank of the correct relation
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank, penalizes low ranks
- **Hits@K**: Percentage of correct relations ranked in top K (K = 1, 3, 5, 10)

## Installation

### Prerequisites
- Python 3.9
- CUDA-compatible GPU (optional but recommended)
- Conda or pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/ShallowBKGC.git
cd ShallowBKGC
```

### Step 2: Create conda environment
```bash
conda create -n shallowbkgc python=3.9 -y
conda activate shallowbkgc

```

Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Dataset Preparation
Place your datasets in the KGs/ folder with the following structure:

text
KGs/
├── FB15K/
│   ├── train.txt          # Training triples
│   ├── valid.txt          # Validation triples
│   ├── test.txt           # Test triples
│   └── entity2text.txt    # Entity descriptions (optional)
├── WN18/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── entity2text.txt
└── ...



### Quick Start  
1. Generate entity ID mapping  
```bash
python main.py --dataset KGs/FB15K --num_of_epochs 1
```

2. Create ordered text file (for BERT embeddings)  
```bash
python DateProcess_order.py FB15K
```

3. Generate BERT embeddings
```bash
python DateProcess_npy.py FB15K
```


4. Train and evaluate
```bash
python main.py --dataset KGs/FB15K --num_of_epochs 150 --embedding_dim 50 --batch_size 1000
```


 Command Line Arguments
Argument	Description	Default
```bash
--dataset	Dataset path in KGs/ folder	KGs/FB15K
--embedding_dim	Embedding dimension	50
--num_of_epochs	Number of training epochs	150
--batch_size	Batch size	1000
--input_dropout	Dropout for input embeddings	0.5
--hidden_dropout	Dropout for hidden layers	0.5
--hidden_width_rate	Hidden layer width multiplier	3
--L2reg	L2 regularization strength	0.1
```


Running Experiments
Train on FB15K (full)
```bash
python main.py --dataset KGs/FB15K --num_of_epochs 150 --embedding_dim 50 --batch_size 1000
```

After training, results are saved in the Experiments/ folder with timestamp:  
Experiments/
└── 2026-03-30_14-30-00/
    └── info.log

### Project Structure  
ShallowBKGC/  
├── main.py                 # Main training script  
├── model.py               # ShallowBKGC model definition  
├── helper_classes.py      # Data loading and experiment management  
├── util.py               # Utility functions  
├── DateProcess_order.py   # Create ordered entity text files  
├── DateProcess_npy.py     # Generate BERT embeddings  
├── requirements.txt       # Python dependencies  
├── README.md             # This file  
└── KGs/                  # Dataset folder  
    ├── FB15K/  
    ├── WN18/  
    └── ...  



### Troubleshooting
FileNotFoundError: entityIDx_json not found
Solution: Run main.py first with 1 epoch to generate the mapping:

```bash
python main.py --dataset KGs/YOUR_DATASET --num_of_epochs 1
```


CUDA out of memory
Solution: Reduce batch size or embedding dimension:

```bash
python main.py --dataset KGs/FB15K --batch_size 500 --embedding_dim 32
```


Missing entity2text.txt
Solution: The script will use entity IDs as text (with a warning). For best results, obtain the entity descriptions from the original dataset.

BERT model download issues
Solution: The first run downloads BERT model (400MB). Ensure stable internet connection. Alternatively, download manually and place in cache folder.
