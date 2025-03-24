
import joblib
import librosa
import numpy as np
import os
print("当前工作目录:", os.getcwd())
print("目录下的所有文件:", os.listdir(os.getcwd()))

# 加载训练好的模型

# 指定完整路径
svm_clf = joblib.load(r"C:\Users\lulu\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection\svm_model.pkl")
rf_clf = joblib.load(r"C:\Users\lulu\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection\rf_model.pkl")
meta_clf = joblib.load(r"C:\Users\lulu\PycharmProjects\Sleep-Bruxism-and-Snoring-Detection\stacked_model.pkl")


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
new_audio = r"C:\Users\lulu\Downloads\one_wave.wav"

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


# def extract_features_from_audio(file_path):
#     """
#     从本地音频文件提取 MFCC + 过零率特征。
#     """
#     y, sr = librosa.load(file_path, sr=8000)  # 读取音频，并重采样至 8kHz
#
#     # 提取 MFCC（13 维特征）
#     mfccs = librosa.feature.mfcc(y=y, sr=8000, n_mfcc=13)
#     avg_mfcc = np.mean(mfccs, axis=1)
#
#     # 提取过零率（1 维特征）
#     zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
#     avg_zero_crossing_rate = 10 * np.mean(zero_crossing_rate, axis=1)
#
#     # 组合特征（14 维）
#     feature = np.hstack([avg_mfcc, avg_zero_crossing_rate])
#     return feature
#
# # 测试新的随机音频文件
# new_audio ="C:/Users/xuany/Desktop/snoring_test2.wav"  # 替换为你的音频文件路径
# new_features = extract_features_from_audio(new_audio)
#
# # 仅保留前 10 个最重要的特征
# new_features_reduced = new_features.reshape(1, -1)  # 使用完整 14 维特征
#
#
# # 使用 SVM 和 随机森林 进行预测
# svm_pred = svm_clf.predict(new_features_reduced)
# rf_pred = rf_clf.predict(new_features_reduced)
#
# # 进行 Stacking（融合 SVM 和 随机森林的预测）
# stacked_input = np.column_stack((svm_pred, rf_pred))
# final_pred = meta_clf.predict(stacked_input)
#
# # 输出最终预测结果
# print(f"🔹 SVM prediction: {'snoring' if svm_pred[0] == 1 else 'no snoring'}")
# print(f"🔹 random forest prediction: {'snoring' if rf_pred[0] == 1 else 'no snoring'}")
# print(f"🔹 Stacked model final prediction: {'snoring' if final_pred[0] == 1 else 'no snoring'}")



# import joblib
# import librosa
# import numpy as np
# import os
# from moviepy.editor import VideoFileClip
#
# # 🔹 显示当前工作目录
# print("🔹 当前工作目录:", os.getcwd())
# print("🔹 目录下的所有文件:", os.listdir(os.getcwd()))
#
# # ✅ 1. 检查模型文件是否存在
# model_files = ["svm_model.pkl", "rf_model.pkl", "stacked_model.pkl", "feature_importance.pkl"]
# missing_files = [f for f in model_files if not os.path.exists(f)]
#
# if missing_files:
#     raise FileNotFoundError(f"❌ 缺少以下模型文件: {missing_files}\n请先运行 `train_model.py` 训练并保存模型！")
#
# # ✅ 2. 加载训练好的模型
# svm_clf = joblib.load("svm_model.pkl")
# rf_clf = joblib.load("rf_model.pkl")
# meta_clf = joblib.load("stacked_model.pkl")
# feature_importance = joblib.load("feature_importance.pkl")
#
# # 🔹 获取前 10 个最重要的特征索引
# top_features = np.argsort(feature_importance)[-10:]
# from pydub import AudioSegment  # 用于格式转换
#
# # ✅ 1. 定义 `m4a` 转 `wav` 的函数
# def convert_m4a_to_wav(m4a_path):
#     """
#     将 .m4a 转换为 .wav，并返回新路径。
#     """
#     wav_path = m4a_path.replace(".m4a", ".wav")  # 修改文件扩展名
#     if not os.path.exists(wav_path):  # 避免重复转换
#         audio = AudioSegment.from_file(m4a_path, format="m4a")
#         audio.export(wav_path, format="wav")
#         print(f"✅ 转换完成: {wav_path}")
#     return wav_path  # 返回转换后的 .wav 文件路径
#
# # ✅ 2. 选择输入文件（支持 `.wav` 和 `.m4a`）
# input_path = r"C:\Users\xuany\Desktop\打鼾测试1.m4a"  # 替换为你的音频文件路径
#
# # 如果是 `.m4a`，先转换为 `.wav`
# if input_path.endswith(".m4a"):
#     input_path = convert_m4a_to_wav(input_path)
# # ✅ 3. 从视频提取音频（如果输入是 `.mp4`）
# def extract_audio_from_video(video_path, output_audio_path):
#     """从视频文件中提取音频，并保存为 WAV 文件。"""
#     if not os.path.exists(output_audio_path):  # 避免重复转换
#         video = VideoFileClip(video_path)
#         video.audio.write_audiofile(output_audio_path, codec='pcm_s16le', fps=16000)
#         print(f"✅ 音频提取完成: {output_audio_path}")
#
# # ✅ 4. 从音频文件提取特征
# def extract_features_from_audio(file_path):
#     """从指定的音频文件提取 MFCC + 过零率特征。"""
#     y, sr = librosa.load(file_path, sr=8000)  # 读取音频，并重采样至 8kHz
#     mfccs = librosa.feature.mfcc(y=y, sr=8000, n_mfcc=13)
#     avg_mfcc = np.mean(mfccs, axis=1)
#     zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
#     avg_zero_crossing_rate = 10 * np.mean(zero_crossing_rate, axis=1)
#     return np.hstack([avg_mfcc, avg_zero_crossing_rate])  # 组合 14 维特征
#
# # ✅ 5. 选择输入文件（支持 `.wav` 和 `.mp4`）
# input_path = "C:/Users/xuany/Desktop/snoring_test2.mp4"  # 可以是 MP4 或 WAV
# if input_path.endswith(".mp4"):
#     wav_path = input_path.replace(".mp4", ".wav")
#     extract_audio_from_video(input_path, wav_path)
#     input_path = wav_path  # 更新为 WAV 路径
#
# # ✅ 6. 提取音频特征
# new_features = extract_features_from_audio(input_path)
# new_features_reduced = new_features.reshape(1, -1)  # 确保是 14 维
#
# # ✅ 7. 进行预测
# svm_pred = svm_clf.predict(new_features_reduced)
# rf_pred = rf_clf.predict(new_features_reduced)
# stacked_input = np.column_stack((svm_pred, rf_pred))
# final_pred = meta_clf.predict(stacked_input)
#
# # ✅ 8. 输出预测结果
# print(f"🔹 SVM prediction: {'snoring' if svm_pred[0] == 1 else 'no snoring'}")
# print(f"🔹 Random Forest prediction: {'snoring' if rf_pred[0] == 1 else 'no snoring'}")
# print(f"🔹 Stacked Model final prediction: {'snoring' if final_pred[0] == 1 else 'no snoring'}")
