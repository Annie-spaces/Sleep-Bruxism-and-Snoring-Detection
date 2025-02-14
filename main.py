#version 5
import requests
import librosa
import io
import soundfile as sf
import numpy as np
snoring_MFCCs = np.empty((0, 13))
for i in range(1, 3):=
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
for i in range(1, 3):
    url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/no_snoring/0_0.wav"
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
