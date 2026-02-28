#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature engineering for real-time premovement detection EEG data.

Loads raw subject .mat files, extracts epochs around movement events,
computes ERP, band power and autocorrelation features, trains simple SVM
models, and writes various diagnostic plots.

Created on Thu Sep 14 21:07:35 2023

@author: anzan
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.io
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils import utils

# ---------------------------------------------------------------------------
# utility helpers
# ---------------------------------------------------------------------------

def clear_ipython_namespace() -> None:
    try:
        from IPython import get_ipython
        get_ipython().run_line_magic("reset", "-sf")
    except Exception:
        pass

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_erp_pdf(epochs, labels, ch_names, time_axis, out_path):
    labels = np.asarray(labels).ravel().astype(int)
    c1 = np.flatnonzero(labels == 1)
    c2 = np.flatnonzero(labels == 2)
    ensure_dir(os.path.dirname(out_path))
    with PdfPages(out_path) as pdf:
        for ci, ch in enumerate(ch_names):
            data = epochs[:, :, ci]
            mean_all = data.mean(axis=0); std_all = data.std(axis=0)
            mean1 = data[c1].mean(axis=0) if c1.size else np.zeros_like(mean_all)
            std1 = data[c1].std(axis=0) if c1.size else np.zeros_like(std_all)
            mean2 = data[c2].mean(axis=0) if c2.size else np.zeros_like(mean_all)
            std2 = data[c2].std(axis=0) if c2.size else np.zeros_like(std_all)
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(time_axis, mean1, "m-", linewidth=2.5, label="D key")
            ax.plot(time_axis, mean1+std1, "m--", linewidth=1)
            ax.plot(time_axis, mean1-std1, "m--", linewidth=1)
            ax.plot(time_axis, mean2, "c-", linewidth=2.5, label="L key")
            ax.plot(time_axis, mean2+std2, "c--", linewidth=1)
            ax.plot(time_axis, mean2-std2, "c--", linewidth=1)
            ax.plot(time_axis, mean_all, "k-", linewidth=1.5, label="overall mean")
            ax.plot(time_axis, mean_all+std_all, "k--", linewidth=1)
            ax.plot(time_axis, mean_all-std_all, "k--", linewidth=1)
            ax.set_title(f"Channel: {ch}")
            ax.set_xlabel("Time (s, relative to movement)")
            ax.set_ylabel("Amplitude (�V)")
            ax.axvline(-0.75, color="k", ls="--")
            ax.axvline(0, color="k", ls="--")
            ax.legend(); pdf.savefig(fig); plt.close(fig)

def save_bandpower_pdf(bp_pre, bp_rest, ch_names, out_path):
    freq_labels=["Delta","Theta","Alpha","Beta","Gamma"]
    n_bands=len(freq_labels)
    ensure_dir(os.path.dirname(out_path))
    with PdfPages(out_path) as pdf:
        for ci,ch in enumerate(ch_names):
            m_pre=bp_pre[:,:,ci].mean(axis=0)
            m_rest=bp_rest[:,:,ci].mean(axis=0)
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(m_pre,"g--o",label="premovement")
            ax.plot(m_rest,"r--o",label="rest")
            ax.set_xticks(np.arange(n_bands)); ax.set_xticklabels(freq_labels)
            ax.set_xlim(-0.5,n_bands-0.5)
            ax.set_xlabel("Frequency band"); ax.set_ylabel("Power (V^2/Hz)")
            ax.set_title(f"{ch} band power"); ax.legend()
            pdf.savefig(fig); plt.close(fig)

def lowpass(data,fs,cutoff=8.0):
    sos=signal.ellip(13,0.01,60,Wn=cutoff,btype="lowpass",fs=fs,output="sos")
    return signal.sosfiltfilt(sos,data,axis=0)

# core

def process_mat_file(path, results_dir):
    """Load one subject file and return raw/data and epochs.

    Also computes average‑mastoid and low‑pass versions so the caller
    doesn't have to repeat the same bookkeeping.

    Returns ``(class_info, raw_data, epochs, epochs_am, epochs_am_lpf,
    time_axis, ch_names, fs)``.
    """
    raw = scipy.io.loadmat(path)["o"]
    data = raw["data"].item()
    markers = raw["marker"].item().ravel()
    ch_names = [c[0][0] for c in raw["chnames"].item()]
    fs = int(raw["sampFreq"][0][0])

    class_info = []
    for idx in range(len(markers) - 1):
        if markers[idx] == 0 and markers[idx + 1] in (1, 2):
            window = markers[idx : idx + int(1.5 * fs)]
            diff = np.diff(window.astype(np.float16))
            neg = np.where(diff < 0)[0]
            if neg.size:
                class_info.append([markers[idx + 1], idx + 1, idx + 1 + neg[0], None])
    class_info = np.vstack(class_info) if class_info else np.empty((0, 4))

    epoch_start, epoch_end = -1.5, 0.75
    if class_info.size:
        gaps = (class_info[1:, 1] - class_info[:-1, 2]) / fs
        valid = np.insert(np.where(gaps > abs(epoch_start))[0] + 1, 0, 0)
        class_info = class_info[valid, :]

    epochs, time_axis = utils.extract_epochs(
        data, (class_info[:, 1] / fs).tolist(), fs, epoch_start, epoch_end
    )

    # mastoid re-reference and low-pass filtered versions
    ear_inds = [ch_names.index("A1"), ch_names.index("A2")] if "A1" in ch_names and "A2" in ch_names else []
    if ear_inds:
        avg_mast = np.mean(data[:, ear_inds], axis=1, keepdims=True)
        data_am = data - avg_mast
        epochs_am, _ = utils.extract_epochs(data_am, (class_info[:, 1] / fs).tolist(), fs, epoch_start, epoch_end)
        data_lpf = lowpass(data_am, fs)
        epochs_am_lpf, _ = utils.extract_epochs(data_lpf, (class_info[:, 1] / fs).tolist(), fs, epoch_start, epoch_end)
    else:
        epochs_am = epochs.copy()
        epochs_am_lpf = epochs.copy()

    # save diagnostics for this subject
    subj_name = os.path.basename(path)
    save_erp_pdf(epochs, class_info[:, 0], ch_names, time_axis,
                 os.path.join(results_dir, subj_name, "averageERP_raw_1vs2.pdf"))
    # band power
    premov = np.where((time_axis >= -0.75) & (time_axis <= 0))[0]
    rest = np.where((time_axis >= -1.5) & (time_axis <= -0.75))[0]
    bp_pre = np.stack([
        utils.compute_features(epochs[i, premov, :], fs, "psd")
        for i in range(len(epochs))
    ])
    bp_rest = np.stack([
        utils.compute_features(epochs[i, rest, :], fs, "psd")
        for i in range(len(epochs))
    ])
    save_bandpower_pdf(bp_pre, bp_rest, ch_names,
                       os.path.join(results_dir, subj_name, "averageBP_raw.pdf"))

    save_erp_pdf(epochs_am, class_info[:, 0], ch_names, time_axis,
                 os.path.join(results_dir, subj_name, "averageERP_AMastoid_1vs2.pdf"))
    save_erp_pdf(epochs_am_lpf, class_info[:, 0], ch_names, time_axis,
                 os.path.join(results_dir, subj_name, "averageERP_AMastoid_LPF_1vs2.pdf"))

    return class_info, data, epochs, epochs_am, epochs_am_lpf, time_axis, ch_names, fs


def main():
    clear_ipython_namespace()
    start_time=time.time()
    cwd=os.path.dirname(os.path.abspath(__file__))
    os.chdir(cwd)
    parent=os.path.dirname(cwd)
    raw_dir=os.path.join(parent,"data","raw")
    results_dir=os.path.join(parent,"results")
    ensure_dir(results_dir)
    files=[f for f in os.listdir(raw_dir) if f.endswith(".mat")]
    print("processing",len(files),"files:",files)
    for fi, fname in enumerate(files):
        ci, data, epochs, epochs_am, epochs_am_lpf, t_axis, ch_names, fs = \
            process_mat_file(os.path.join(raw_dir, fname), results_dir)
        # everything has already been saved inside the helper
        # optionally collect for later aggregation
        # (not used here)
        _ = ci, data, epochs, epochs_am, epochs_am_lpf, t_axis, ch_names, fs
    print("finished in",time.time()-start_time,"seconds")

if __name__=="__main__":
    main()
