# Real-time Pre-movement Intention Detection using EEG

## Project Overview

This project implements a real-time system for detecting pre-movement intention from EEG signals using machine learning. The system processes EEG data, extracts relevant features, trains a Support Vector Machine (SVM) classifier, and provides a graphical user interface (GUI) for real-time prediction and visualization.

The methodology involves:
- EEG signal preprocessing (average mastoid re-referencing)
- Feature extraction including Event-Related Potentials (ERP), Power Spectral Density (PSD), Autocorrelation, and Low-Pass Filtered ERP
- SVM classification with hyperparameter tuning
- Real-time streaming and prediction through a PyQt5-based GUI

## Features

- **EEG Data Processing**: Load and process .mat files containing EEG signals and markers
- **Feature Engineering**: Extract multiple types of features from EEG epochs
- **Model Training**: Train and evaluate SVM models with different feature combinations
- **Real-time Detection**: GUI for loading EEG data and performing live predictions
- **Visualization**: Plot EEG signals and prediction results over time
- **Data Streaming**: Simulate real-time data streaming with pause/continue functionality

## Installation

### Prerequisites

- Python 3.7+
- Required packages (install via pip):

```bash
pip install numpy scipy joblib PyQt5 matplotlib scikit-learn pandas
```

## Usage

### 1. Feature Engineering and Model Training

Run `feature_engineering_v01.py` to:
- Process raw EEG data files
- Extract and visualize features
- Train SVM models with different feature combinations
- Save the best performing model

```bash
python feature_engineering_v01.py
```

This script will:
- Load .mat files from `data/raw/`
- Perform average mastoid re-referencing
- Extract epochs around event markers
- Compute features (ERP, PSD, Autocorrelation, LPFERP)
- Train linear SVM with various feature combinations
- Save the best model to `results/models/best_linear_SVM.pkl`
- Generate visualization PDFs in `results/`

### 2. Real-time GUI Application

Run `implementation_GUI.py` to launch the GUI:

```bash
python implementation_GUI.py
```

The GUI provides:
- **Load EEG Signal**: Select a .mat file for analysis
- **EEG Plot**: Real-time visualization of EEG channels
- **Prediction Display**: True vs Predicted labels over time
- **Controls**: Pause/Continue streaming, Exit

The system streams through the EEG data in windows, extracts features, and predicts pre-movement intention using the trained SVM model.

### 3. Utility Functions

`utils.py` contains helper functions for:
- Epoch extraction from continuous EEG data
- Feature computation (ERP, PSD, Autocorrelation, Entropy)
- Feature combination extraction
- Data import from .dat/.csv files

## Authors

- Bhoj Raj Thapa