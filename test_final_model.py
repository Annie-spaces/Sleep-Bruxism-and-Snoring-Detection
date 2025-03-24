import joblib
import librosa
import numpy as np
import os
print("当前工作目录:", os.getcwd())
print("目录下的所有文件:", os.listdir(os.getcwd()))

# 加载训练好的模型

# 指定完整路径
svm_clf = joblib.load(r"C:\Users\xuany\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection-new\svm_model.pkl")
rf_clf = joblib.load(r"C:\Users\xuany\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection-new\rf_model.pkl")
meta_clf = joblib.load(r"C:\Users\xuany\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection-new\stacked_model.pkl")


# 加载特征重要性，并选择前 10 个最重要的特征
feature_importance = joblib.load("feature_importance.pkl")
top_features = np.argsort(feature_importance)[-10:]

def segment_and_extract_features(audio_path, sr=8000, clip_duration=1.0, step_duration=0.5):
    y, orig_sr = librosa.load(audio_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    clip_len = int(sr * clip_duration)
    step_len = int(sr * step_duration)

    features = []
    for start in range(0, len(y_resampled) - clip_len + 1, step_len):
        clip = y_resampled[start: start + clip_len]

        mfccs = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13)
        avg_mfcc = np.mean(mfccs, axis=1)

        zcr = librosa.feature.zero_crossing_rate(clip)
        avg_zcr = 10 * np.mean(zcr, axis=1)

        feature_vector = np.hstack([avg_mfcc, avg_zcr])
        features.append(feature_vector)

    return np.array(features)

# 测试音频路径
new_audio = "C:/Users/xuany/Desktop/snoring_test2.wav"

# 切片并提取所有片段的特征
clips_features = segment_and_extract_features(new_audio)  # shape: (num_clips, 14)
print(f"cut clips get {clips_features.shape[0]} 个片段，every clips has 14 features")

# 对每个片段预测
for i, feature in enumerate(clips_features):
    feature_input = feature.reshape(1, -1)

    svm_pred = svm_clf.predict(feature_input)
    rf_pred = rf_clf.predict(feature_input)
    stacked_input = np.column_stack((svm_pred, rf_pred))
    final_pred = meta_clf.predict(stacked_input)

    print(f"part {i+1:02d} second ➤ {'Snoring' if final_pred[0] == 1 else 'No Snoring'}")