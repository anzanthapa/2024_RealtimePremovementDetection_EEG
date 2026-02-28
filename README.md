# Real-time Pre-movement Intention Detection using EEG

## Project Overview

This repository implements a complete pipeline for detecting a subject's
intention to move before the actual motion onset, using scalp EEG
recordings.  The workflow is divided into three major stages:

1. **Data preprocessing and epoching** – raw `.mat` files are parsed,
   average-mastoid re-referenced and segmented around movement-related
   markers.
2. **Feature engineering & model building** – epochs are converted into
   ERP waveforms, band‑power (PSD), autocorrelation patterns and low‑pass
   filtered ERP.  A linear SVM is trained over all feature-set
   combinations to identify the best predictor.
3. **Interactive demonstration** – a PyQt5 GUI streams stored EEG
   recordings in windowed chunks, extracts features on the fly and
   displays real-time predictions from the trained classifier.

Key characteristics:

- Offline training with persistent diagnostic plots (ERP / band power)
- Simple, pipeline-friendly code structure for easy modification
- Lightweight real-time simulator supporting pause/continue control

## Features

- **Modular preprocessing**
  - epoch extraction with adjustable start/end times
  - automatic mastoid re-referencing and optional low-pass filtering
- **Multiple feature modalities**
  - raw ERP traces
  - frequency‑band powers (Delta–Gamma)
  - autocorrelation profiles
  - low‑pass ERP after 8 Hz filtering
- **Model evaluation**
  - exhaustive search over four binary feature inclusion flags
  - linear SVM with train/validation split and accuracy reporting
  - best pipeline saved to disk for later use
- **GUI streaming**
  - selectable input file dialog
  - scrolling EEG plot with channel labels
  - prediction widget showing true vs. forecast labels
  - simple data stream abstraction for fixed-size windows

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anzanthapa/2024_RealtimePremovementDetection_EEG.git
   cd 2024_RealtimePremovementDetection_EEG
   ```
2. Create and activate a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(alternatively: `pip install numpy scipy joblib PyQt5 matplotlib scikit-learn pandas`)*

## Usage

### 1. Run preprocessing & training

- Ensure raw `.mat` files are placed under `data/raw/`.
- Execute
  ```bash
  python feature_engineering.py
  ```
- The script will:
  * iterate over all subjects/files in the raw directory
  * compute epochs and the four feature sets for each epoch
  * build and evaluate a linear SVM over every feature combination
  * save the best-performing factory pipeline to
    `results/models/best_linear_SVM.pkl`
  * dump diagnostic PDFs (`averageERP_*.pdf`, `averageBP_raw.pdf`)
    under `results/<subject>/`

### 2. Launch the GUI

- Run the graphical demo:
  ```bash
  python implementation_GUI.py
  ```
- Controls:
  * **Load EEG Signal** – choose a `.mat` dataset
  * **Pause / Continue** – stop/resume the streaming iterator
  * **Exit** – close the application
- The application processes the data in 150‑sample windows and feeds
  features to the stored SVM, displaying predictions alongside true
  labels.

### 3. Utilities

- `utils.py` exports:
  * `extract_epochs()` – window slicing helper
  * `compute_features()` – handles ERP/PSD/autocorr/entropy
  * `extract_features()` – builds combined feature vectors
  * `importdata()` – generic CSV reader with transpose

These can be imported wherever custom processing is required.

## Authors

- Bhoj Raj Thapa (GitHub: **anzanthapa**)

## Citation

If you reference or build upon this work, please acknowledge the
repository as follows:

```bibtex
@misc{thapa2024_realtime_eeg_pmid,
  author = {Thapa, Bhoj Raj},
  title = {Real-time Pre-movement Intention Detection using EEG},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/anzanthapa/2024_RealtimePremovementDetection_EEG}}
}
```

Greater detail and attribution information can be found in the source
headers.
