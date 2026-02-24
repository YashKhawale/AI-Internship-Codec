# ==========================================
# SPAM EMAIL CLASSIFIER
# Detects spam or ham messages using NLP
# ==========================================

# Add the SMSSpamCollection file present in data folder inside the /contents/sample_data while running on Google Collab

import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("\nLoading Dataset...")

try:
    df = pd.read_csv("/content/sample_data/SMSSpamCollection", sep='\t', names=["label", "message"])
except Exception as e:
    print(f"ERROR: Could not load dataset. Make sure 'SMSSpamCollection' is in the correct path.")
    print(f"Details: {e}")
    print("Dataset link: https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip")

if 'df' in locals() and not df.empty:
    # Text Cleaning
    def clean_text(text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        return text

    df["message"] = df["message"].apply(clean_text)

    df["label"] = df["label"].map({"ham":0, "spam":1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42
    )

    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])

    print("Training Model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions))

    print("\nType messages to test (type 'exit' to quit)\n")

    while True:
        msg = input("Enter Message: ")
        if msg.lower() == "exit":
            break

        result = model.predict([msg])[0]

        if result == 1:
            print("Prediction: SPAM ❌\n")
        else:
            print("Prediction: NOT SPAM ✅\n")
elif 'df' not in locals():
    print("DataFrame 'df' was not loaded. Please resolve the dataset path issue and re-run the cell.")

