
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        y, sr = sf.read(io.BytesIO(response.content),dtype="float32")  # Read the WAV file
        if len(y.shape) == 2:  # Convert to mono if stereo
            y = librosa.to_mono(y.T)
        y_new = librosa.resample(y, orig_sr=sr, target_sr=16000)  # Resample to 16kHz
        return(y_new)
    else:
        raise Exception(f"Failed to download: {url}")


### Function to Extract Features (MFCC + Zero-Crossing Rate)
def extract_features(y_new):
        # Extract Features
        mfccs = librosa.feature.mfcc(y=y_new, sr=16000, n_mfcc=13, n_fft=400, hop_length=200) # 13 MFCCs
        avg_mfcc = np.mean(mfccs, axis=1)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_new, frame_length=400, hop_length=200)
        avg_zero_crossing_rate = np.mean(zero_crossing_rate, axis=1)  # Scaled for uniformity

        # Combine Features
        feature = np.hstack([avg_mfcc, avg_zero_crossing_rate])  # 14 Features
        feature = feature.reshape(1, -1)
        feature_reduced=feature[:,[6,5,13,1,8,3,4,12,9]]
        return feature_reduced



# Load Snoring Dataset
number_of_features=9
number_of_total_samples=500
number_of_customized_samples=6
snoring_features = np.empty((0, number_of_features))
for i in range(number_of_total_samples):  # 500 snoring samples
    url = f"https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/snoring/1_{i}.wav"
    audio=get_audio_from_url(url)
    # audio = audio / np.max(np.abs(audio))  # Normalize to -1.0 to 1.0
    snoring_features = np.vstack([snoring_features, extract_features(audio)])

# Load Non-Snoring Dataset
no_snoring_features = np.empty((0, number_of_features))
for i in range(number_of_total_samples-number_of_customized_samples):  # 500 non-snoring samples
    url = f"https://raw.githubusercontent.com/adrianagaler/Snoring-Detection/master/Snoring_Dataset_%4016000/no_snoring/0_{i}.wav"
    audio = get_audio_from_url(url)
    # if np.max(np.abs(audio))!=0:
        # audio = audio / (np.max(np.abs(audio)))  # Normalize to -1.0 to 1.0
    no_snoring_features = np.vstack([no_snoring_features, extract_features(audio)])

print("shape is", no_snoring_features.shape)
#load customised backgroung noise files
for i in range(1,1+number_of_customized_samples):
    y, sr = sf.read(f"C:\\Users\\lulu\\Downloads\\background{i}.wav", dtype="float32")  # Read the WAV file
    if len(y.shape) == 2:  # Convert to mono if stereo
        y = librosa.to_mono(y.T)
    y_new = librosa.resample(y, orig_sr=sr, target_sr=16000)  # Resample to 16kHz
    no_snoring_features = np.vstack([no_snoring_features, extract_features(y_new)])

# Combine Data & Create Labels (1 = Snoring, 0 = Non-Snoring)
X = np.vstack([snoring_features, no_snoring_features])
y = np.array([1] * number_of_total_samples + [0] * number_of_total_samples)  # Labels

# feature normalization
print(f"min is {np.min(X, axis=0)}")
print(f"max is {np.max(X, axis=0)}")
min_max = MinMaxScaler()
X = min_max.fit_transform(X)

# setup training Dataset (80% Train, 20% Test)
X_train, X_test_and_val, y_train, y_test_and_val = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test_and_val, y_test_and_val, test_size=0.5, random_state=10)

####### Train & Evaluate Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=9) # Initialize Random Forest with 100 trees
rf_clf.fit(X_train, y_train)
y_final_pred_rf_val = rf_clf.predict(X_val)# Make predictions on the test set
y_final_pred_rf_train = rf_clf.predict(X_train)

y_pred_rf_val = rf_clf.predict_proba(X_val)# Make predictions on the test set
y_pred_rf_train = rf_clf.predict_proba(X_train)

# print("forest probability", y_pred_rf_val)
print("\n🔹 Random Forest Classification Report on train:")
print(classification_report(y_train, y_final_pred_rf_train))
print("\n🔹 Random Forest Classification Report on val:")
print(classification_report(y_val, y_final_pred_rf_val))

feature_importance = rf_clf.feature_importances_  # Get importance scores for all 14 features

#Train & Evaluate SVM
svm_clf = SVC(kernel="rbf", C=0.6, gamma="scale")
# X_train=np.hstack((X_train, y_pred_rf_train[:, 1].reshape(-1, 1)))

X_val_svm=np.hstack((X_val, y_pred_rf_val[:, 1].reshape(-1, 1)))
svm_clf.fit(X_val_svm, y_val)

# X_val_svm=np.hstack((X_val, y_pred_rf_val[:, 1].reshape(-1, 1)))
y_pred_svm_val = svm_clf.predict(X_val_svm)

print("🔹 SVM Classification Report on val:")
print(classification_report(y_val, y_pred_svm_val))

# 然后评估测试集
rf_test_preds = rf_clf.predict_proba(X_test)
rf_test_final_preds = rf_clf.predict(X_test)
X_test_svm=np.hstack((X_test, rf_test_preds[:, 1].reshape(-1, 1)))
svm_test_preds = svm_clf.predict(X_test_svm)
# stacked_test = np.hstack((X_test, svm_test_preds.reshape(-1,1), rf_test_preds.reshape(-1,1)))

# final_preds = meta_clf.predict(stacked_test)

print("\n🔹 Random Forest Classification Report:")
print(classification_report(y_test, rf_test_final_preds))

print("\n🔹 Cascaded SVM Classification Report:")
print(classification_report(y_test, svm_test_preds))

# Compute the confusion matrix
cm = confusion_matrix(y_test, svm_test_preds)
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

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
joblib.dump(svm_clf, "svm_model_v3.pkl")  # 保存 SVM 模型
joblib.dump(rf_clf, "rf_model_v3.pkl")  # 保存随机森林模型
# 保存特征重要性（如果使用随机森林选出的特征）
joblib.dump(feature_importance, "feature_importance.pkl")

print("✅ 训练完成，模型和特征重要性已保存！")

