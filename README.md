
# Sentiment Analysis on Video & Text Data
---

## 1. Project Overview
This project performs **sentiment analysis** on multimodal data consisting of **video clips** and **textual utterances**.  
Each utterance from a video is classified into one of three categories: **Positive**, **Negative**, or **Neutral**.

The project demonstrates the integration of **text processing**, **video frame analysis**, and **machine learning techniques** to build a robust sentiment classification system.

---

## 2. Dataset Description

- **Text Data:** CSV files containing utterances, speaker information, sentiment labels, dialogue ID, episode, season, start and end times.  
- **Video Data:** Corresponding video clips for each utterance stored in directories for train and test sets.  
- **Sample Columns:** `Utterance`, `Speaker`, `Sentiment`, `Dialogue_ID`, `Utterance_ID`, `Season`, `Episode`, `StartTime`, `EndTime`.

**Example:**

| Sr No. | Utterance                                     | Speaker         | Sentiment | Dialogue_ID | Utterance_ID | Season | Episode | StartTime    | EndTime      |
| ------ | --------------------------------------------- | --------------- | --------- | ----------- | ------------ | ------ | ------- | ------------ | ------------ |
| 4      | So let’s talk a little bit about your duties. | The Interviewer | neutral   | 0           | 3            | 8      | 21      | 00:16:26,820 | 00:16:29,572 |
| 18     | No, I-I-I-I don't, I actually don't know      | Rachel          | negative  | 1           | 3            | 9      | 23      | 00:36:49,290 | 00:36:51,791 |

---

## 3. Project Workflow & Implementation

The system consists of **six major modules**:

### 3.1 Loading and Processing Data
- Load CSV files for train and test datasets.
- Each row is linked to its **corresponding video clip** using `Dialogue_ID` and `Utterance_ID`.

### 3.2 Text Feature Engineering
- **Preprocessing steps:** Lowercasing, removing special characters, lemmatization, and stopwords removal using NLTK.
- Extracted **text-based features**:
  - Word count
  - Character count
  - Sentiment polarity
- Applied **Word2Vec embeddings** for richer semantic representation.

### 3.3 Time-Based Features
- `StartTime` and `EndTime` converted to seconds.
- Calculated **duration of utterances** to provide temporal context.

### 3.4 Video Processing and Frame Extraction
- **Frames extracted** at intervals from each video clip.
- Converted frames to **grayscale** and extracted features:
  - Mean, standard deviation, median, min-max difference.
- Feature vectors **averaged across frames** for each video.

### 3.5 Data Merging and Preparation
- Merged **text features**, **video features**, and **audio features** based on the video file name.
- Each sample is represented as a **single concatenated feature vector**.

### 3.6 Sentiment Classification
- **Model used:** Gradient Boosting Classifier (with hyperparameter tuning using GridSearchCV).
- **Training & Validation:**
  - Split: 80% train, 20% validation.
  - Label encoding for sentiment classes.
- **Performance Metrics:**
  - Accuracy
  - Classification report
  - Confusion matrix
- **Test Predictions:** Saved as `submission.csv`.

---

## 4. Technology Stack

| Component                  | Description                                                        |
| -------------------------- | ------------------------------------------------------------------ |
| **Language**               | Python 3                                                           |
| **Libraries**              | Numpy, Pandas, OpenCV, Scikit-learn, NLTK, Matplotlib, Seaborn     |
| **Machine Learning Model** | Gradient Boosting Classifier                                       |
| **Text Processing**        | Lemmatization, Stopword Removal, Word2Vec Embeddings               |
| **Video Processing**       | Frame Extraction, Feature Calculation (mean, std, median, min-max) |

---

## 5. Implementation Details / Functions

### 5.1 Video Feature Extraction
The system extracts features from video clips by processing a set number of frames per video. For each frame:

- Frames are sampled uniformly across the video duration.
- Each frame is converted to grayscale.
- Statistical features are computed from the grayscale frame: mean intensity, standard deviation, median intensity, and min-max difference.
- The feature vectors from all frames are averaged to produce a single feature vector representing the video clip.

This approach reduces the dimensionality while preserving key visual information relevant to sentiment.

### 5.2 Text Preprocessing
Textual utterances undergo a series of preprocessing steps to clean and normalize the data:

- Convert all text to lowercase.
- Remove non-alphabetic characters to eliminate noise.
- Perform **lemmatization** to reduce words to their root forms.
- Remove **stopwords** using the NLTK stopword list to focus on meaningful words.

After preprocessing, the text is vectorized using a **Bag-of-Words model** (CountVectorizer) and optionally enhanced using **Word2Vec embeddings** for semantic representation.

### 5.3 Audio/Time Feature Extraction
Audio or time-based features are extracted to provide additional context:

- `StartTime` and `EndTime` are converted to seconds.
- The **duration** of each utterance is computed.
- Basic audio statistics can be computed from the video if needed, such as frame rate, number of frames, and derived metrics like frames per second or total duration.

These features allow the model to incorporate temporal information about the utterance.

### 5.4 Data Merging and Preparation
The extracted **text, video, and audio features** are merged for each sample:

- Feature vectors are aligned using the video file identifier.
- All relevant features are concatenated into a **single feature vector** for each utterance.
- This merged feature set serves as input for the machine learning model.

### 5.5 Sentiment Classification
The final classification is performed using a **Gradient Boosting Classifier**:

- Labels are encoded numerically (Positive, Negative, Neutral).
- The dataset is split into training and validation sets (80/20 split).
- Hyperparameter tuning is done using **GridSearchCV** to optimize model parameters like learning rate, depth, and number of estimators.
- The trained model predicts sentiment for validation and test sets.
- Performance is evaluated using **accuracy**, **classification reports**, and **confusion matrices**.

This approach effectively combines multimodal features to produce robust sentiment predictions.

---

## 6. Results and Evaluation

- **Validation Accuracy:** ~ Depends on your run
- **Confusion Matrix:** Visualized using Seaborn heatmap
- **Feature Importance:** Extracted from Gradient Boosting model

**Sample Output in `submission.csv`:**

| Sr No. | Sentiment |
| ------ | --------- |
| 62     | neutral   |
| 72     | neutral   |
| 112    | neutral   |
| 120    | positive  |
| 318    | negative  |

---

## 7. Challenges & Considerations

- Handling **large video files** and frame extraction efficiently
- Feature alignment between **text and video modalities**
- Dealing with **imbalanced sentiment classes** in some datasets
- Optimizing **hyperparameters** to improve model generalization

---
