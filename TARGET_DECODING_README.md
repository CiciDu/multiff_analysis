# Target Decoding from Monkey Neural Data

This repository contains code to decode a monkey's internal representation of targets from neural activity data. The implementation uses multiple approaches including Canonical Correlation Analysis (CCA) and machine learning methods to understand how targets are encoded in neural populations.

## 🧠 Overview

**Target decoding** aims to predict what target a monkey is thinking about or pursuing based solely on neural activity patterns. This is important for understanding:

- How spatial information is encoded in neural populations
- Which neurons are most important for target representation
- Temporal dynamics of target encoding
- Neural correlates of behavioral planning

## 📁 Files Description

### Core Implementation
- **`target_decoder.py`** - Main decoder class with CCA and ML methods
- **`decode_targets_example.py`** - Standalone script demonstrating all functionality
- **`notebooks/target_decoding_example.ipynb`** - Interactive notebook tutorial

### Existing Codebase (methods/non_behavioral_analysis/neural_data_analysis/decode_targets/)
- **`decode_target_class.py`** - Core data processing class
- **`decode_target_utils.py`** - Utility functions for data preparation
- **`behav_features_to_keep.py`** - Configuration for behavioral features
- **`fit_gpfa_utils.py`** - GPFA (Gaussian Process Factor Analysis) utilities
- **`cca_class.py`** - Canonical Correlation Analysis implementation

## 🚀 Quick Start

### 1. Basic Usage

```python
from target_decoder import TargetDecoder

# Initialize decoder
decoder = TargetDecoder("path/to/monkey/data")

# Load and prepare data
decoder.load_and_prepare_data()

# Perform CCA-based decoding
cca_results = decoder.decode_with_cca()

# Decode specific target properties
ml_results = decoder.decode_target_variable('target_distance')

# Visualize results
decoder.plot_cca_results()
```
=
### 2. Run Complete Analysis

```bash
python decode_targets_example.py
```

This will:
1. Load neural and behavioral data
2. Perform CCA analysis 
3. Train machine learning models
4. Generate visualizations and save results

## 📊 Methods Implemented

### 1. Canonical Correlation Analysis (CCA)
- Finds linear combinations of neural activity maximally correlated with target features
- Identifies which neural patterns relate to target representation
- Provides interpretable loadings showing neuron contributions

### 2. Machine Learning Decoding
- **Random Forest Regression** - Non-linear ensemble method
- **Ridge Regression** - Linear method with regularization
- Cross-validation for robust performance estimation
- Feature importance analysis

### 3. Data Preprocessing
- Temporal binning of spike data (default: 20ms bins)
- Lag inclusion for temporal dynamics
- Missing value handling
- Feature standardization

## 📈 Interpretation Guide

### CCA Results
- **Canonical Correlations**: Strength of neural-target relationships (0-1)
- **Neural Loadings**: Which neurons contribute to each component
- **Target Loadings**: Which target features are represented
- **Canonical Variables**: Low-dimensional projections

### ML Results  
- **R² Score**: Proportion of variance explained (higher = better)
  - R² > 0.3: Strong decoding
  - R² > 0.1: Moderate decoding
  - R² < 0.1: Weak decoding
- **Feature Importance**: Which neurons are most predictive
- **Cross-validation**: Generalization performance

## 🔧 Configuration Options

### Time Binning
```python
decoder = TargetDecoder(
    raw_data_folder_path="path/to/data",
    bin_width=0.02,      # 20ms time bins
    window_width=0.05    # 50ms sliding window
)
```

### CCA Parameters
```python
cca_results = decoder.decode_with_cca(
    n_components=10      # Number of canonical components
)
```

### ML Parameters
```python
results = decoder.decode_target_variable(
    target_var='target_distance',
    model_type='rf',     # 'rf' or 'ridge'
    test_size=0.2        # Train/test split
)
```

## 📋 Target Variables

The decoder can predict various target properties:

**Spatial Properties:**
- `target_x`, `target_y` - Target coordinates
- `target_distance` - Distance to target
- `target_angle` - Angle to target
- `target_rel_x`, `target_rel_y` - Relative position

**Temporal Properties:**
- `time_since_target_last_seen` - Time since target visibility
- `target_last_seen_distance` - Distance when last seen

**Behavioral Properties:**
- `target_visible_dummy` - Target visibility status
- `target_angle_to_boundary` - Angle relative to boundary

## 🔍 Example Output

```
MONKEY TARGET REPRESENTATION DECODING
============================================================

Step 1: Initializing decoder...
✓ Decoder initialized

Step 2: Loading neural and behavioral data...
✓ Data loaded successfully
  Neural data shape: (33125, 242)
  Target data shape: (33125, 115)
  Time points: 33,125
  Neurons: 242
  Recording duration: ~11.0 minutes

Step 3: Performing Canonical Correlation Analysis...
✓ CCA completed
  Top 5 canonical correlations: [0.68786 0.43551 0.40293 0.27123 0.25526]
  Strongest neural-target correlation: 0.688
  Variance explained by 1st component: 47.3%

Step 4: Machine learning-based target decoding...
✓ Successfully decoded target_distance
  Cross-validation R²: 0.425 ± 0.032
  Test R²: 0.441

SUMMARY:
• CCA found 10 canonical components
• Strongest correlation: 0.688
• ML decoding R²: 0.441
• Strong decoding performance achieved
```

## 📊 Visualization Outputs

The code generates several plots:

1. **CCA Results** (`cca_results.png`)
   - Canonical correlations bar plot
   - Neural and target loadings heatmaps
   - Canonical variables scatter plot

2. **ML Results** (`ml_results_[variable].png`)
   - Predicted vs true values
   - Residuals analysis

3. **Feature Importance** (`feature_importance_[variable].png`)
   - Top contributing neurons
   - Importance rankings

## ⚙️ Requirements

```python
numpy >= 1.19.0
pandas >= 1.3.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
```

**Neuroscience packages:**
```python
elephant >= 0.10.0  # For GPFA analysis
neo >= 0.9.0        # Neural data structures
quantities >= 0.12.0
```

## 🔧 Troubleshooting

### Common Issues

**1. Data Loading Errors**
```
Error: No such file or directory
```
- Check that data path is correct
- Ensure processed data files exist (run preprocessing first)

**2. Low Decoding Performance**
```
R² < 0.1
```
- Try different target variables
- Increase temporal window
- Check data quality (firing rates, target variability)

**3. Memory Issues**
```
MemoryError: Unable to allocate array
```
- Reduce number of time lags
- Subsample data temporally
- Use fewer neurons (feature selection)

**4. CCA Convergence**
```
CCA failed: Convergence error
```
- Check for NaN/infinite values
- Reduce number of components
- Normalize data

## 📚 Scientific Background

### Canonical Correlation Analysis
CCA finds linear combinations of two sets of variables (neural activity and target features) that are maximally correlated. In neuroscience, this reveals:
- Population-level neural patterns
- Relationships between brain and behavior
- Dimensionality of neural representations

### Machine Learning Decoding
ML approaches predict behavioral variables from neural activity:
- **Random Forest**: Captures non-linear relationships
- **Ridge Regression**: Linear decoder with regularization
- **Cross-validation**: Prevents overfitting

### Feature Importance
Identifies which neurons contribute most to target encoding:
- Important for understanding neural circuits
- Guides electrode placement in experiments
- Reveals functional organization

## 🔬 Applications

1. **Brain-Computer Interfaces**: Decode intended movements
2. **Neural Prosthetics**: Control external devices
3. **Cognitive Neuroscience**: Understand spatial cognition
4. **Clinical Applications**: Assess neural function

## 📖 References

- Cunningham, J.P. & Yu, B.M. (2014). Dimensionality reduction for large-scale neural recordings. *Nature Neuroscience*, 17(11), 1500-1509.
- Hardcastle, K., Maheswaranathan, N., Ganguli, S., & Giocomo, L.M. (2017). A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex. *Neuron*, 94(2), 375-387.
- Stringer, C., Pachitariu, M., Steinmetz, N., Reddy, C.B., Carandini, M., & Harris, K.D. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. *Science*, 364(6437), eaav7893.

## 👥 Contributing

To contribute to this codebase:
1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Submit a pull request

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review existing issues
- Create a new issue with detailed description

---

*This implementation is part of the multiff_analysis project for studying monkey behavioral and neural data.* 