# Variational Autoencoders for Multi-Modal Hybrid-Language Music Clustering

A comprehensive implementation of VAE-based music clustering with multi-modal fusion (audio + lyrics) across 5 languages.

## Project Overview

This project implements multiple VAE architectures for unsupervised clustering of music:
- **Languages**: Arabic, Bangla, English, Hindi, Spanish
- **Dataset**: 1,120 audio files, 677 with lyrics
- **Models**: Basic VAE, Conv-VAE, Beta-VAE, CVAE, VaDE
- **Fusion**: Early, Late, and Attention-based multi-modal fusion

## Performance Optimizations

- **GPU**: Mixed precision training with automatic GPU utilization
- **CPU**: 15-core parallel data loading and preprocessing
- **Memory**: Efficient batch processing with gradient accumulation

## Features

### Audio Processing
- Mel-spectrogram extraction (128 bands, 22.05kHz)
- MFCC features (40 coefficients)
- Audio augmentation (pitch shift, time stretch, noise injection)

### Lyrics Processing
- Multilingual transformer embeddings (XLM-RoBERTa)
- Original, Romanized, and Translated lyrics extraction
- Support for 5 languages with automatic language detection

### VAE Architectures
1. **Basic VAE**: Fully connected encoder-decoder
2. **Convolutional VAE**: 2D CNN for spectrograms
3. **Beta-VAE**: Disentangled representations (beta=4)
4. **Conditional VAE**: Genre/language conditioning
5. **VaDE**: Variational Deep Embedding with GMM priors

### Clustering Methods
- K-Means with multiple initializations
- Agglomerative Hierarchical Clustering
- DBSCAN (density-based)
- Gaussian Mixture Models

### Evaluation Metrics
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Cluster Purity

## Project Structure

```
project/
├── data/
│   ├── audio/              # 1,120 audio files by language
│   ├── processed_lyrics/   # 677 lyrics files
│   └── features/           # Extracted features cache
├── src/
│   ├── models/
│   │   ├── vae.py          # Basic VAE
│   │   ├── conv_vae.py     # Convolutional VAE
│   │   ├── beta_vae.py     # Beta-VAE
│   │   ├── cvae.py         # Conditional VAE
│   │   └── vade.py         # VaDE
│   ├── data/
│   │   ├── dataset.py      # PyTorch datasets
│   │   ├── audio_processor.py     # Audio feature extraction
│   │   ├── lyrics_processor.py    # Lyrics extraction & embedding
│   │   └── data_matcher.py        # Audio-lyrics pairing
│   ├── clustering/
│   │   ├── cluster.py      # Clustering algorithms
│   │   └── evaluation.py   # Metrics computation
│   ├── fusion/
│   │   └── multimodal.py   # Fusion strategies
│   ├── training/
│   │   ├── trainer.py      # Training loop
│   │   └── config.py       # Configuration
│   └── visualization/
│       └── plots.py        # Visualization utilities
├── experiments/
│   ├── train_vae.py        # Training scripts
│   ├── run_clustering.py   # Clustering experiments
│   └── baseline.py         # Baseline comparisons
├── results/
│   ├── checkpoints/
│   ├── visualizations/
│   └── metrics/
├── notebooks/
│   └── exploratory.ipynb   # Data exploration
├── configs/
│   └── config.yaml         # Experiment configs
├── requirements.txt
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Data Preprocessing

```bash
# Extract audio features (uses all 1,120 audio files)
python src/data/audio_processor.py --n_workers 15

# Process lyrics (uses 677 lyric files)
python src/data/lyrics_processor.py --multilingual

# Match audio-lyrics pairs
python src/data/data_matcher.py
```

### 2. Training VAE Models

```bash
# 1. Basic VAE (Easy Task) - Audio only
python experiments/train_vae.py --model basic --modality audio

# 2. Convolutional VAE (Medium Task) - Audio only
python experiments/train_vae.py --model conv --modality audio

# 3. Convolutional VAE - Multi-modal (audio + lyrics)
python experiments/train_vae.py --model conv --modality multimodal

# 4. Beta-VAE (Hard Task) - Audio only
python experiments/train_vae.py --model beta --modality audio --beta 4.0

# 5. Conditional VAE - Genre conditioning (45 genres)
python experiments/train_vae.py --model cvae --modality audio --condition genre

# 6. Conditional VAE - Language conditioning (5 languages)
python experiments/train_vae.py --model cvae --modality audio --condition language

# 7. VaDE - Clustering with GMM priors (50 clusters)
python experiments/train_vae.py --model vade --modality audio --n_clusters 50
```

**Directory Structure:**
- Each model saves to unique directory: `results/checkpoints/{model_name}/`
- Model names: `basic`, `conv`, `conv_multimodal`, `beta`, `cvae_genre`, `cvae_language`, `vade`

### 3. Evaluate Models

```bash
python evaluate_results.py --model basic
python evaluate_results.py --model conv
python evaluate_results.py --model conv_multimodal
python evaluate_results.py --model beta
python evaluate_results.py --model cvae_genre
python evaluate_results.py --model cvae_language
python evaluate_results.py --model vade
```

### 4. Clustering

```bash
# Run clustering experiments
python experiments/run_clustering.py --model conv_vae --checkpoint best_model.pt

# Compare with baselines
python experiments/baseline.py
```

### 4. Visualization

```bash
# Generate all visualizations
python src/visualization/plots.py --model all --output results/visualizations/
```

### 5. Model Testing & Feature Proof

**Verify Dataset Structure:**
```bash
# Confirm 5 languages, 45 genres, and audio-lyrics matching
python verify_dataset.py
```

**Test Individual Model Features:**
```bash
# Test 1: Basic VAE - Smooth latent space interpolation
python test_models.py --test smoothness --model basic_vae

# Test 2: Conv VAE - Learned convolutional filters
python test_models.py --test filters --model conv_vae

# Test 3: Beta-VAE - Disentanglement (each dimension = independent factor)
python test_models.py --test disentanglement --model beta_vae

# Test 4: Conditional VAE - Class separation (language or genre)
python test_models.py --test separation --model cvae --condition language
python test_models.py --test separation --model cvae --condition genre

# Test 5: VaDE - Soft clustering with probability scores
python test_models.py --test soft_clustering --model vade
```

**Compare All Models:**
```bash
# Run all tests and generate comparison report
python test_models.py --test all --output_dir results/model_tests/
```

**Quick Training Scripts:**
```bash
# Quick test (trains Conv VAE for 50 epochs, ~2-3 hours)
python quick_start.py

# Full training pipeline (all 5 models, ~20-24 hours)
python train_all.py
```

**Expected Test Results:**
- Basic VAE: Smooth interpolation with consistent step distances (~0.002)
- Conv VAE: 32 learned filters detecting frequency bands and rhythms
- Beta-VAE: Independent factor control (tempo, pitch, energy, genre)
- CVAE (language): 5 well-separated clusters (ARI > 0.8)
- CVAE (genre): 45 genre clusters with hierarchical structure (ARI > 0.6)
- VaDE: Soft cluster assignments with confidence scores (mean > 0.85)

See [MODEL_COMPARISON.md](MODEL_COMPARISON.md) and [ARCHITECTURE_PROOF.md](ARCHITECTURE_PROOF.md) for detailed explanations.

## Task Coverage

### Easy Task
- Basic VAE on audio features
- K-Means clustering
- Comparison with PCA + K-Means
- Metrics: Silhouette Score, CH Index

### Medium Task
- Convolutional VAE on spectrograms
- Multi-modal (audio + lyrics)
- Multiple clustering algorithms
- Metrics: Silhouette, DB Index, ARI

### Hard Task
- Beta-VAE and CVAE
- Multi-modal with attention fusion
- All clustering algorithms
- All 6 evaluation metrics
- Extensive visualizations

## Results

Results are saved in `results/` directory:
- `checkpoints/`: Trained model weights
- `metrics/`: CSV files with all metrics
- `visualizations/`: Latent space plots, cluster distributions, reconstructions

## Performance Tips

1. **GPU Memory**: Adjust batch size based on VRAM (default: 32)
2. **CPU Cores**: Set `--n_workers 15` for maximum parallelization
3. **Mixed Precision**: Enabled by default, reduces memory by 50%
4. **Feature Caching**: Features are cached after first extraction

## Citation

If you use this code, please cite:

```bibtex
@article{mostakim2026vae,
  title={Variational Autoencoders for Multi-Modal Hybrid-Language Music Clustering},
  author={Mostakim, Moin},
  year={2026}
}
```

## License

MIT License

## Contact

Moin Mostakim
Course: Neural Networks
Submission: January 10th, 2026
#   4 2 5 _ p r o j e c t  
 