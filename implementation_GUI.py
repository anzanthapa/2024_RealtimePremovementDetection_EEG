# -*- coding: utf-8 -*-
"""
Simple PyQt GUI for loading EEG data, streaming windows and displaying
model predictions.  Used to demo the self-paced pre-movement
intention detector visually.

Created on Fri Mar 29 00:10:01 2024

@author: anzanthapa
"""

import os
import numpy as np
import scipy.io
import joblib
from scipy import signal
from utils import utils

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel,
    QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QIcon

CHANNEL_NAMES = [
    "FP1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
    "A1","A2","F7","F8","T3","T4","T5","T6","Fz","Cz","Pz","X5"
]
BAD_CHANNELS = {"A1","A2","X5"}
VALID_CHANNEL_INDICES = [i for i,ch in enumerate(CHANNEL_NAMES) if ch not in BAD_CHANNELS]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.pardir, "results", "models", "best_linear_SVM.pkl")

class EEGVisualizer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Self-paced Pre-movement Intention Detector")
        self.setGeometry(100, 100, 1000, 800)

        # configuration constants
        self.best_combo = [0, 0, 0, 1]

        self.load_model()
        self.init_ui()
        self.default_plot()

    def load_model(self) -> None:
        """Lazy‑load the trained classifier pipeline from disk."""
        self.pipeline = joblib.load(MODEL_PATH)

    def init_ui(self):
        app_dir=os.path.join(os.path.dirname(__file__),os.pardir,"app")
        icon=QIcon(os.path.join(app_dir,"ICON.png"))
        self.setWindowIcon(icon)
        QApplication.setWindowIcon(icon)
        layout=QVBoxLayout()
        btn_layout=QHBoxLayout()
        self.loadButton=QPushButton("Load EEG Signal")
        self.loadButton.clicked.connect(self.load_eeg)
        btn_layout.addWidget(self.loadButton)
        self.pauseButton=QPushButton("Pause")
        self.pauseButton.clicked.connect(self.pause_stream)
        btn_layout.addWidget(self.pauseButton)
        self.continueButton=QPushButton("Continue")
        self.continueButton.clicked.connect(self.continue_stream)
        btn_layout.addWidget(self.continueButton)
        self.exitButton=QPushButton("Exit")
        self.exitButton.clicked.connect(self.close)
        btn_layout.addWidget(self.exitButton)
        layout.addLayout(btn_layout)
        self.predictionWidget=PredictionWidget(self)
        layout.addWidget(self.predictionWidget)
        self.eegPlot=EEGPlot(self,width=8,height=4)
        layout.addWidget(self.eegPlot)
        container=QWidget(); container.setLayout(layout)
        self.setCentralWidget(container)
        self.timer=QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.default_style="QPushButton {}"
        self.active_style="QPushButton { background-color: green; color: white; }"

    def default_plot(self):
        data=np.zeros((150,len(VALID_CHANNEL_INDICES)))
        labels=[CHANNEL_NAMES[i] for i in VALID_CHANNEL_INDICES]
        self.eegPlot.plot(data,labels,0,150)

    def load_eeg(self) -> None:
        """Prompt user for file and initialise streaming state."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load EEG File", os.path.join(os.pardir, "data", "raw")
        )
        if not path:
            print("No file selected")
            return
        ds = scipy.io.loadmat(path)["o"]
        self.eeg_data = ds["data"].item()
        self.ch_names = [c[0][0] for c in ds["chnames"].item()]
        self.sample_rate = ds["sampFreq"][0][0].item()
        self.raw_markers = ds["marker"].item()
        self.modified_markers = np.where((self.raw_markers == 1) | (self.raw_markers == 2), 1, 0)

        self.data_stream = EEGDataStream(self.eeg_data, self.modified_markers)
        self.refresh_time = 250
        self.timer.start(self.refresh_time)
        print("File loaded successfully.")

    def update_plot(self) -> None:
        """Fetch next window, redraw plot and issue a prediction."""
        if not hasattr(self, "data_stream"):
            print("Data is not loaded.")
            return
        try:
            data_segment, marker_segment, start_idx, end_idx = next(
                self.data_stream.get_next_window()
            )
        except StopIteration:
            self.timer.stop()
            print("End of detection.")
            return

        labels = [CHANNEL_NAMES[i] for i in VALID_CHANNEL_INDICES]
        self.eegPlot.plot(data_segment[:, VALID_CHANNEL_INDICES], labels, start_idx, end_idx)
        true_label = marker_segment[-1] == 1

        test_features = utils.extract_features(
            data_segment,
            200,
            CHANNEL_NAMES,
            include_erp=self.best_combo[0],
            include_PSD=self.best_combo[1],
            include_AC=self.best_combo[2],
            include_LPFERP=self.best_combo[3],
        )
        pred_label = self.pipeline.predict(test_features)
        self.predictionWidget.updatePrediction(true_label, pred_label, start_idx, end_idx)

    def pause_stream(self) -> None:
        self.timer.stop()
        self.pauseButton.setStyleSheet(self.active_style)
        self.continueButton.setStyleSheet(self.default_style)
        print("Detection paused.")

    def continue_stream(self) -> None:
        self.timer.start(self.refresh_time)
        self.pauseButton.setStyleSheet(self.default_style)
        self.continueButton.setStyleSheet(self.default_style)
        print("Detection continued.")


# ---------------------------------------------------------------------------
# additional classes needed by the GUI
# ---------------------------------------------------------------------------

class PredictionWidget(QWidget):
    """Shows true/predicted labels and epoch indices."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.true_label = QLabel("True: ?")
        self.pred_label = QLabel("Pred: ?")
        self.index_label = QLabel("0-0")
        l = QHBoxLayout()
        l.addWidget(self.true_label)
        l.addWidget(self.pred_label)
        l.addWidget(self.index_label)
        self.setLayout(l)

    def updatePrediction(self, true: bool, pred: np.ndarray, start: int, end: int):
        self.true_label.setText(f"True: {int(true)}")
        self.pred_label.setText(f"Pred: {int(pred)}")
        self.index_label.setText(f"{start}-{end}")


class EEGPlot(FigureCanvas):
    """Simple scrolling EEG plot using Matplotlib."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.axes = fig.add_subplot(111)
        self.lines = []

    def plot(self, data, labels, start_idx, end_idx):
        # data: samples x channels
        self.axes.clear()
        nchan = data.shape[1]
        t = np.arange(start_idx, end_idx)
        for ci in range(nchan):
            self.axes.plot(t, data[:, ci] + ci * 100, label=labels[ci])
        self.axes.set_ylim(-100, nchan * 100 + 100)
        self.axes.set_xlim(start_idx, end_idx)
        self.draw()


class EEGDataStream:
    """Iterator that yields fixed-length windows from EEG data."""

    def __init__(self, eeg_data, markers, window_size=150, step=1):
        self.eeg = eeg_data
        self.markers = markers
        self.win = window_size
        self.step = step
        self.idx = 0
        self.n = eeg_data.shape[0]

    def get_next_window(self):
        while self.idx + self.win <= self.n:
            seg = self.eeg[self.idx : self.idx + self.win, :]
            mseg = self.markers[self.idx : self.idx + self.win]
            start = self.idx
            end = self.idx + self.win
            self.idx += self.step
            yield seg, mseg, start, end
        raise StopIteration()

