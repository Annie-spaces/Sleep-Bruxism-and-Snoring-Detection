import joblib
import librosa
import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf

svm_clf = joblib.load(r"C:\Users\lulu\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection\svm_model_v3.pkl")
rf_clf = joblib.load(r"C:\Users\lulu\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection\rf_model_v3.pkl")

def extract_features(clip, sr):
    mfccs = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13, n_fft=400, hop_length=200)
    avg_mfcc = np.mean(mfccs, axis=1)
    zcr = librosa.feature.zero_crossing_rate(clip, frame_length=400, hop_length=200)
    avg_zcr = 10 * np.mean(zcr, axis=1)
    # print("average zerpo crossing rate for the second is", avg_zcr)
    # print("average mfcc is", avg_mfcc)
    feature_vector = np.hstack([avg_mfcc, avg_zcr])
    # print("the feature vector is ", feature_vector, "with shape", feature_vector.shape)
    # normalization
    minimum = np.array([-35.92499801, -37.62589326, 0., -102.43443268, -50.3469832,
                        -76.67146314, -49.91350851, -33.78452263, -31.04088984])
    maximum = np.array([62.37346823, 65.67767637, 6.18179012, 215.87052958, 29.63762666,
                        64.38628459, 53.21305143, 62.59366706, 61.50408267])
    noramlized_reduced_feature_vector=(feature_vector[[6,5,13,1,8,3,4,12,9]]-minimum)/(maximum-minimum)

    # print(f"noramlized_reduced_feature_vector is {noramlized_reduced_feature_vector}")
    return noramlized_reduced_feature_vector

def check_snoring(feature):
    feature_input = feature.reshape(1, -1)
    cascaded_input=np.hstack((feature_input, rf_clf.predict_proba(feature_input)[:, 1].reshape(-1, 1)))
    svm_pred = svm_clf.predict(cascaded_input)
    return(svm_pred)


FORMAT = pyaudio.paFloat32
CHANNELS = 1
# print("original rate is", params.SAMPLE_RATE, f"now is 16000")
RATE = 16000
WIN_SIZE_SEC = 1
CHUNK = int(WIN_SIZE_SEC * RATE)
RECORD_SECONDS = 9*60*60

audio = pyaudio.PyAudio()
# start Recording
stream = audio.open(format=FORMAT,
                    # input_device_index=MIC,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
print("recording...")
try:
    for i in range(0, RECORD_SECONDS):
        print(f"the {i+1} second")
        # Read 1 second of audio
        audio = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        print("captured audio",audio)
        print("with shape", audio.shape)
        # Normalize to range -1.0 to 1.0 if needed
        # audio = audio / np.max(np.abs(audio))  # Normalize to -1.0 to 1.0
        audio = np.clip(audio*1000, -1.0, 1.0)
        clip_features = extract_features(audio, RATE)
        # print("shape of feature is",clip_features.shape)
        snore=check_snoring(clip_features)
        if snore==True:
            print("a snore!")
            filename = r"C:\Users\lulu\Downloads\1_second_tone.mp3"
            data, samplerate = sf.read(filename)
            sd.play(data, samplerate)
except KeyboardInterrupt:
    print("\n🛑 Stopping real-time processing...")

# Stop recording
print("✅ Processing finished.")
stream.stop_stream()
stream.close()
audio.terminate()