import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import cv2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import re
import warnings
from nltk.corpus import stopwords

# Ensure you have downloaded the stopwords if not already done
import nltk
nltk.download('wordnet')

nltk.download('stopwords')

# Define basic_stopwords using NLTK stopwords
basic_stopwords = set(stopwords.words('english'))


warnings.filterwarnings('ignore')

# %% Data Loading and Initial Processing
# Load datasets
train_df = pd.read_csv('/Users/shaswethashankar/PycharmProjects/Machine_Learning_Lab/HACK/Set_3/train/text.csv',
                       encoding='ISO-8859-1')
test_df = pd.read_csv('/Users/shaswethashankar/PycharmProjects/Machine_Learning_Lab/HACK/Set_3/test/text.csv',
                      encoding='ISO-8859-1')

# Define paths to video folders
train_video_dir = '/Users/shaswethashankar/PycharmProjects/Machine_Learning_Lab/HACK/Set_3/train/videos'
test_video_dir = '/Users/shaswethashankar/PycharmProjects/Machine_Learning_Lab/HACK/Set_3/test/videos'

# Function to get video file path
def get_video_clip_path(row, video_dir):
    dialogue_id = row['Dialogue_ID']
    utterance_id = row['Utterance_ID']
    filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
    return os.path.join(video_dir, filename)

# Create video paths
train_df['video_clip_path'] = train_df.apply(lambda row: get_video_clip_path(row, train_video_dir), axis=1)
test_df['video_clip_path'] = test_df.apply(lambda row: get_video_clip_path(row, test_video_dir), axis=1)

# %% Text Processing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in basic_stopwords]
    return ' '.join(words)

# Apply text preprocessing
train_df['processed_text'] = train_df['Utterance'].apply(preprocess_text)
test_df['processed_text'] = test_df['Utterance'].apply(preprocess_text)

# %% Feature Extraction Functions
def extract_video_features(video_path, num_frames=10):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros(40)
        features = []
        frame_indices = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            features.extend([np.mean(gray), np.std(gray), np.median(gray), np.max(gray) - np.min(gray)])
        cap.release()
        if len(features) < 40:
            features.extend([0] * (40 - len(features)))
        return np.array(features[:40])
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        return np.zeros(40)

def extract_audio_features(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros(10)
        sample_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / sample_rate
        features = [sample_rate, frame_count, duration, sample_rate * frame_count, frame_count / duration if duration > 0 else 0]
        cap.release()
        if len(features) < 10:
            features.extend([0] * (10 - len(features)))
        return np.array(features[:10])
    except Exception as e:
        print(f"Error extracting audio features from {video_path}: {str(e)}")
        return np.zeros(10)

# %% Extract Features
print("Extracting text features...")
vectorizer = CountVectorizer(max_features=100)
train_text_features = vectorizer.fit_transform(train_df['processed_text']).toarray()
test_text_features = vectorizer.transform(test_df['processed_text']).toarray()

print("Extracting video features...")
train_video_features = np.array([extract_video_features(path) for path in train_df['video_clip_path']])
test_video_features = np.array([extract_video_features(path) for path in test_df['video_clip_path']])

print("Extracting audio features...")
train_audio_features = np.array([extract_audio_features(path) for path in train_df['video_clip_path']])
test_audio_features = np.array([extract_audio_features(path) for path in test_df['video_clip_path']])

train_features = np.hstack([train_text_features, train_video_features, train_audio_features])
test_features = np.hstack([test_text_features, test_video_features, test_audio_features])

# %% Model Training with Hyperparameter Tuning
print("Training model...")
le = LabelEncoder()
train_labels = le.fit_transform(train_df['Sentiment'])

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7]
}

grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Validate model
val_predictions = model.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, val_predictions))
print("\nClassification Report:")
print(classification_report(le.inverse_transform(y_val), le.inverse_transform(val_predictions)))

# %% Model Evaluation and Visualization
print("Feature Importance Visualization...")
plt.figure(figsize=(12, 6))
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.title('Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

print("Confusion Matrix...")
cm = confusion_matrix(y_val, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %% Make Predictions and Create Submission
print("Making predictions on test set...")
test_predictions = model.predict(test_features)
test_predictions = le.inverse_transform(test_predictions)

submission_df = pd.DataFrame({
    'Sr No.': test_df['Sr No.'],
    'Sentiment': test_predictions
})
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file created!")