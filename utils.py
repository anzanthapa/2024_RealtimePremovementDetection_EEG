# -*- coding: utf-8 -*-
"""
Collection of helper functions used across the project.  Includes epoch
extraction, feature computation (ERP, PSD, autocorr, entropy) and a
general-purpose file importer.

Created on Thu Sep 21 19:46:07 2023

@author: anzanthapa
"""

import numpy as np
import pandas as pd
import warnings
from scipy import signal, stats

def IMAEEG(eeg_data):
    """Placeholder for integrated moving average EEG feature.

    The original project defined this function but its implementation was
    never provided.  Calling it will raise an error so the user is reminded
    to implement or remove it.
    """
    raise NotImplementedError("IMAEEG is not implemented")


def extract_epochs(continuous_signal, event_markers, fs, relative_epoch_start, relative_epoch_end):
    if not isinstance(continuous_signal, np.ndarray):
        raise ValueError("continuous_signal should be a 2D or 1D NumPy array.")
    if continuous_signal.ndim == 1:
        continuous_signal = continuous_signal.reshape(-1,1)
    if isinstance(event_markers, np.ndarray) and event_markers.ndim == 1:
        event_markers = event_markers.tolist()
    num_channels = continuous_signal.shape[1]
    if fs is None:
        fs = 1
    time_axis = np.arange(relative_epoch_start*fs, relative_epoch_end*fs)/fs
    num_epochs = len(event_markers)
    num_samples = int((relative_epoch_end-relative_epoch_start)*fs)
    epochs = np.zeros((num_epochs, num_samples, num_channels))
    for i,e in enumerate(event_markers):
        start = int(round(e*fs) + round(relative_epoch_start*fs))
        end = int(round(e*fs) + round(relative_epoch_end*fs) -1)
        if start>=0 and end < continuous_signal.shape[0]:
            epochs[i,:,:] = continuous_signal[start:end+1,:]
    return epochs, time_axis


def compute_features(eeg_epoch, fs, feature_type):
    if not isinstance(eeg_epoch, np.ndarray):
        raise TypeError("eeg_epoch must be ndarray")
    if eeg_epoch.ndim ==1:
        eeg_epoch = eeg_epoch.reshape(-1,1)
    nch = eeg_epoch.shape[1]
    if feature_type.lower()=='erp':
        return eeg_epoch
    if feature_type.lower()=='psd':
        mat=np.empty((5,nch))
        for chi in range(nch):
            freq,psd = signal.welch(eeg_epoch[:,chi],fs=fs,nperseg=fs//2)
            mat[:,chi]=[
                np.sum(psd[(freq>=0)&(freq<=4)]),
                np.sum(psd[(freq>4)&(freq<=8)]),
                np.sum(psd[(freq>8)&(freq<=13)]),
                np.sum(psd[(freq>13)&(freq<=30)]),
                np.sum(psd[freq>30])
            ]
        return mat
    if feature_type.lower()=='autocorr':
        out=np.empty(eeg_epoch.shape)
        for chi in range(nch):
            sig=eeg_epoch[:,chi]-eeg_epoch[:,chi].mean()
            ac=signal.correlate(sig,sig,method='direct')
            out[:,chi]=ac[eeg_epoch.shape[0]-1:]/(np.var(eeg_epoch[:,chi])*len(eeg_epoch[:,chi]))
        return out
    if feature_type.lower()=='entropy':
        out=np.empty((1,nch))
        for chi in range(nch):
            out[0,chi]=stats.entropy(eeg_epoch[:,chi])
        return out
    raise ValueError("Unsupported feature type")


def extract_features(eeg_epoch_raw, fs, ch_names,
                     include_erp=True, include_PSD=True,
                     include_AC=True, include_LPFERP=True):
    ear_inds=[ch_names.index('A1'),ch_names.index('A2')]
    mast=np.mean(eeg_epoch_raw[:,ear_inds],axis=1).reshape(-1,1)
    eeg=eeg_epoch_raw-mast
    feats=[]
    if include_erp:
        chans=['F3','F4','C3','C4','P3','P4','F7','T3','T4','Fz','Cz','Pz']
        idx=[ch_names.index(c) for c in chans if c in ch_names]
        erp=compute_features(eeg,fs,'erp')
        feats.append(erp[:,idx].flatten().reshape(1,-1))
    if include_PSD:
        chans=['FP1','Fp2','F3','F4','C3','C4','P3','P4','O1','02','F7','T3','T4','Fz','Cz','Pz']
        idx=[ch_names.index(c) for c in chans if c in ch_names]
        psd=compute_features(eeg,fs,'psd')
        feats.append(psd[:,idx].flatten().reshape(1,-1))
    if include_AC:
        chans=['FP1','Fp2','F4','C3','C4','P3','P4','O1','02','F7','F8','T3','T5','Fz','Cz','Pz']
        idx=[ch_names.index(c) for c in chans if c in ch_names]
        ac=compute_features(eeg,fs,'autocorr')
        feats.append(ac[:,idx].flatten().reshape(1,-1))
    if include_LPFERP:
        chans=['F3','F4','C3','C4','P3','P4','F7','T3','T4','Fz','Cz','Pz']
        idx=[ch_names.index(c) for c in chans if c in ch_names]
        sos=signal.ellip(13,0.01,60,Wn=8,btype='lowpass',fs=fs,output='sos')
        lp=np.zeros((eeg.shape[0],len(idx)))
        for i,chi in enumerate(idx):
            lp[:,i]=signal.sosfiltfilt(sos,eeg[:,chi])
        feats.append(lp.flatten().reshape(1,-1))
    return np.hstack(feats)


def importdata(file_path,delimiter=None,header=None):
    try:
        df=pd.read_csv(file_path,delimiter=delimiter,header=header)
    except Exception as e:
        print(f"error: {e}")
        return None,None
    data=df.values.T
    names=df.columns.values if df.columns.size>0 else np.array([])
    return data,names
