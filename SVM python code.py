import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# wav_url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_0.wav"
# response = requests.get(wav_url)
# if response.status_code == 200:
#     y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
#     y_new = librosa.resample(y, orig_sr=sr, target_sr=8000) #resampled y
#     sr_new=8000 #new sampling rate
#     print("Original Audio Shape:", y.shape)
#     print("Resampled Audio Shape:", y_new.shape)
# else:
#     raise Exception("Failed to download the audio file")
# mfccs = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
# print(mfccs)
# print("MFCC Shape:", mfccs.shape)

# 加载数据
#version 5
import requests
import librosa
import io
import soundfile as sf
import numpy as np
snoring_MFCCs = np.empty((0, 13))
for i in range(1, 101):
    url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_" + str(i) + ".wav"
    response = requests.get(url)
    if response.status_code == 200:
        y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
        y_new = librosa.resample(y, orig_sr=sr, target_sr=8000)  # resampled y
        sr_new = 8000  # new sampling rate
    else:
        raise Exception("Failed to download the audio file")
    mfccs = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
    average_mfcc = np.mean(mfccs, axis=1)
    snoring_MFCCs=np.vstack([snoring_MFCCs, average_mfcc])
#
no_snoring_MFCCs=np.empty((0, 13))
for i in range(1, 101):
    url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/no_snoring/0_"+ str(i) + ".wav"
    response = requests.get(url)
    if response.status_code == 200:
        print("I'm getting it")
        y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
        if len(y.shape) == 2:  # Check if it's stereo
            y = librosa.to_mono(y.T)  # Convert to mono
        y_new = librosa.resample(y, orig_sr=sr, target_sr=8000)  # resampled y
        sr_new = 8000  # new sampling rate
    else:
        raise Exception("Failed to download the audio file")
    mfccs = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
    average_mfcc = np.mean(mfccs, axis=1)
    print('for', i, '.wav, average mfcc is', average_mfcc)
    no_snoring_MFCCs=np.vstack([no_snoring_MFCCs, average_mfcc])
print('snoring_MFCCs')
print(snoring_MFCCs)
print('no_snoring_MFCCs')
print( no_snoring_MFCCs)

# 数据集

X=np.vstack([snoring_MFCCs,no_snoring_MFCCs])
#audio_files = snore_files + non_snore_files
labels = [1] * 100 + [0] * 100
print(labels)
 # 1代表鼾声, 0代表非鼾声
y=np.array(labels)
# 提特征
# X, y = load_audio_features(audio_files, labels)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练SVM
clf = SVC(kernel="rbf", C=1.0, gamma="scale")  # RBF核适用于复杂数据
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 测评一下
print(classification_report(y_test, y_pred, target_names=["Non-Snore", "Snore"]))



