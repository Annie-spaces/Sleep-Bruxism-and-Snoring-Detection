#version 1, local memory, original frames
# import librosa
# file_path = r"C:\Users\lulu\Downloads\human_voice.wav" # Replace with your file path
# y, sr = librosa.load(file_path, sr=None)  # Load with original sampling rate
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# print(mfccs)

# Plotting MFCC
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis="time", sr=sr)
# plt.colorbar()
# plt.title("MFCC Features")
# plt.xlabel("Time")
# plt.ylabel("MFCC Coefficients")
# plt.show()
# print("MFCC Shape:", mfccs.shape)  # (Number of MFCCs, Time Frames)


#verison 2, online url, original frame
# import requests
# import librosa
# import io
# import soundfile as sf
# wav_url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_0.wav"
# response = requests.get(wav_url)
# if response.status_code == 200:
#     y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
#     print('sample rate is', sr)
# else:
#     raise Exception("Failed to download the audio file")
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
# print(mfccs)

#verison 3, online url, resampled frame
import requests
import librosa
import io
import soundfile as sf
# wav_url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/no_snoring/0_1.wav"
# response = requests.get(wav_url)
# if response.status_code == 200:
#     y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
#     if len(y.shape) == 2:  # Check if it's stereo
#         y = librosa.to_mono(y.T)  # Convert to mono
#     y_new = librosa.resample(y, orig_sr=sr, target_sr=8000) #resampled y
#     sr_new=8000 #new sampling rate
#     print("Original Audio Shape:", y.shape)
#     print("Resampled Audio Shape:", y_new.shape)
# else:
#     raise Exception("Failed to download the audio file")
# mfccs = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
# print(mfccs)
# print("MFCC Shape:", mfccs.shape)

#version 4 reading multiple online url, getting average mfcc for each sample, store them in a numpy array (column is features, row is data)
# import numpy as np
# snoring_MFCCs = np.empty((0, 13))
# for i in range(1, 3):
#     url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_" + str(i) + ".wav"
#     response = requests.get(url)
#     if response.status_code == 200:
#         y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
#         y_new = librosa.resample(y, orig_sr=sr, target_sr=8000)  # resampled y
#         sr_new = 8000  # new sampling rate
#     else:
#         raise Exception("Failed to download the audio file")
#     mfccs = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
#     average_mfcc = np.mean(mfccs, axis=1)
#     snoring_MFCCs=np.vstack([snoring_MFCCs, average_mfcc])

# no_snoring_MFCCs=np.empty((0, 13))
# for i in range(1, 3):
#     url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/no_snoring/0_0.wav"
#     response = requests.get(url)
#     if response.status_code == 200:
#         print("I'm getting it")
#         y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
#         if len(y.shape) == 2:  # Check if it's stereo
#             y = librosa.to_mono(y.T)  # Convert to mono
#         y_new = librosa.resample(y, orig_sr=sr, target_sr=8000)  # resampled y
#         sr_new = 8000  # new sampling rate
#     else:
#         raise Exception("Failed to download the audio file")
#     mfccs = librosa.feature.mfcc(y=y_new, sr=sr_new, n_mfcc=13)
#     average_mfcc = np.mean(mfccs, axis=1)
#     print('for', i, '.wav, average mfcc is', average_mfcc)
#     no_snoring_MFCCs=np.vstack([no_snoring_MFCCs, average_mfcc])
# print('snoring_MFCCs')
# print(snoring_MFCCs)
# print('no_snoring_MFCCs')
# print( no_snoring_MFCCs)

#version 5
import numpy as np
snoring_MFCCs = np.empty((0, 13))
for i in range(1, 3):
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




