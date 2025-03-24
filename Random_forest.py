
### Import Libraries
import numpy as np
import librosa
import soundfile as sf
import requests
import io
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler

### Function to Extract Features (MFCC + Zero-Crossing Rate)
def extract_features(url):
    """Downloads and extracts MFCC + Zero Crossing Rate features from an audio file."""
    response = requests.get(url)
    if response.status_code == 200:
        y, sr = sf.read(io.BytesIO(response.content))  # Read the WAV file
        if len(y.shape) == 2:  # Convert to mono if stereo
            y = librosa.to_mono(y.T)
        y_new = librosa.resample(y, orig_sr=sr, target_sr=8000)  # Resample to 8kHz

        # Extract Features
        mfccs = librosa.feature.mfcc(y=y_new, sr=8000, n_mfcc=13)  # 13 MFCCs
        avg_mfcc = np.mean(mfccs, axis=1)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_new, frame_length=2048, hop_length=512)
        avg_zero_crossing_rate = 10 * np.mean(zero_crossing_rate, axis=1)  # Scaled for uniformity

        # Combine Features
        feature = np.hstack([avg_mfcc, avg_zero_crossing_rate])  # 14 Features

        return feature

    else:
        raise Exception(f"Failed to download: {url}")


# Load Snoring Dataset
number_of_total_samples=500
snoring_features = np.empty((0, 14))
for i in range(number_of_total_samples):  # 500 snoring samples
    url = f"https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_{i}.wav"
    snoring_features = np.vstack([snoring_features, extract_features(url)])

# Load Non-Snoring Dataset
no_snoring_features = np.empty((0, 14))
for i in range(number_of_total_samples):  # 500 non-snoring samples
    url = f"https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/no_snoring/0_{i}.wav"
    no_snoring_features = np.vstack([no_snoring_features, extract_features(url)])

# Combine Data & Create Labels (1 = Snoring, 0 = Non-Snoring)
X = np.vstack([snoring_features, no_snoring_features])
y = np.array([1] * number_of_total_samples + [0] * number_of_total_samples)  # Labels

# feature normalization
min_max = MinMaxScaler()
X = min_max.fit_transform(X)

# setup training Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train & Evaluate SVM
svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

print("🔹 SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

####### Train & Evaluate Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42) # Initialize Random Forest with 100 trees
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)# Make predictions on the test set

print("\n🔹 Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

####### Feature Selection (Use Top Features from RF in SVM)
feature_importance = rf_clf.feature_importances_  # Get importance scores for all 14 features
top_features = np.argsort(feature_importance)[-10:]  # Select top 10 important features

X_train_reduced = X_train[:, top_features] #reduce the dataset to only the top 10 selected features
X_test_reduced = X_test[:, top_features]

svm_clf_reduced = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_clf_reduced.fit(X_train_reduced, y_train)
y_pred_svm_reduced = svm_clf_reduced.predict(X_test_reduced)

print("\n🔹 SVM (Reduced Features) Classification Report:")
print(classification_report(y_test, y_pred_svm_reduced))

# 用训练集做 stacking 特征
svm_train_preds = svm_clf.predict(X_train)
rf_train_preds = rf_clf.predict(X_train)
stacked_train = np.column_stack((svm_train_preds, rf_train_preds))

meta_clf = LogisticRegression()
meta_clf.fit(stacked_train, y_train)

# 然后评估测试集
svm_test_preds = svm_clf.predict(X_test)
rf_test_preds = rf_clf.predict(X_test)
stacked_test = np.column_stack((svm_test_preds, rf_test_preds))

final_preds = meta_clf.predict(stacked_test)

print("\n🔹 Stacked Model Classification Report:")
print(classification_report(y_test, final_preds))

##check which features rd pick
# Get feature importance scores from the trained Random Forest model
feature_importance = rf_clf.feature_importances_

# Sort feature indices based on importance (highest to lowest)
sorted_indices = np.argsort(feature_importance)[::-1]  # Descending order

# Print feature importance values
print("🔹 Feature Importance Rankings (Higher is more important):")
for i, index in enumerate(sorted_indices):
    print(f"Feature {index}: Importance Score = {feature_importance[index]:.4f}")


# 保存训练好的 SVM、随机森林 和 Stacked 模型
joblib.dump(svm_clf, "svm_model.pkl")  # 保存 SVM 模型
joblib.dump(rf_clf, "rf_model.pkl")  # 保存随机森林模型
joblib.dump(meta_clf, "stacked_model.pkl")  # 保存 Stacking 模型（逻辑回归）
# 保存特征重要性（如果使用随机森林选出的特征）
joblib.dump(feature_importance, "feature_importance.pkl")

print("✅ 训练完成，模型和特征重要性已保存！")

