import requests
import librosa
import io
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
snoring_MFCCs = np.empty((0, 13))



# for i in range(0, 1):
#     url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_" + str(i) + ".wav"
#     response = requests.get(url)
#     if response.status_code == 200:
#         y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
#         y_new = librosa.resample(y, orig_sr=sr, target_sr=8000)  # resampled y
#         sr_new = 8000  # new sampling rate
#     else:
#         raise Exception("Failed to download the audio file")



for i in range(0, 1):
    url = "https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/no_snoring/0_" + str(i) + ".wav"
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
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_new, frame_length=2048, hop_length=512, center=True)
    print('mfccs')
    print(mfccs)
    print('')
    print('zero crossing rate')
    print(zero_crossing_rate)

    zero_crossing_rate = zero_crossing_rate.flatten()  # (16,)

    # Create a figure and axis for plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the zero crossing rate
    ax1.plot(zero_crossing_rate, label="Zero Crossing Rate", color='tab:blue')

    # Set labels and title
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Zero Crossing Rate', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Zero Crossing Rate across Frames')

    # Show the plot
    plt.tight_layout()
    plt.show()

    #MFCC plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis="time", sr=sr_new)
    plt.colorbar()
    plt.title("MFCC Features")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()
    print("MFCC Shape:", mfccs.shape)  # (Number of MFCCs, Time Frames)