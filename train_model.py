# ==========================================
# FINAL Fake News Detector Training Script
# ==========================================

import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# LOAD DATA
# ===============================
print("Loading dataset...")

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]]
df.dropna(inplace=True)

print("Dataset size:", df.shape)

# ===============================
# CLEAN TEXT
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ===============================
# PIPELINE (TF-IDF + Calibrated SVM)
# ===============================
print("Building model...")

base_model = LinearSVC()

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=8000,
        ngram_range=(1,2),
        stop_words="english"
    )),
    ("model", CalibratedClassifierCV(base_model))
])

# ===============================
# TRAIN
# ===============================
print("Training model...")
pipeline.fit(X_train, y_train)

# ===============================
# EVALUATE
# ===============================
print("\nEvaluating...")

pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(pipeline, "model.pkl")

print("\n✅ model.pkl saved successfully")
print("🎉 Training Completed!")