import pandas as pd

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from pickle import dump


# ============================================================
# Load spaCy
# ============================================================

print("Loading spaCy model...")

nlp = spacy.load(
    "en_core_web_lg"
)

print("spaCy model loaded.")


# ============================================================
# Text cleaning function
# ============================================================

def clean_function(text):

    if not text:

        return ""


    # Lowercase

    text = text.lower()


    # spaCy

    doc = nlp(text)


    cleaned_tokens = []


    for token in doc:

        # Remove punctuation

        if token.is_punct:
            continue


        # Remove stop words

        if token.is_stop:
            continue


        # Lemmatization

        cleaned_tokens.append(
            token.lemma_
        )


    return " ".join(
        cleaned_tokens
    )


# ============================================================
# Load dataset
# ============================================================

print("Loading spam.csv...")

data = pd.read_csv(
    "spam.csv"
)


# ============================================================
# Check dataset
# ============================================================

print()
print("Dataset shape:")
print(data.shape)

print()

print("Missing values:")
print(data.isnull().sum())


# ============================================================
# Remove missing values
# ============================================================

data = data.dropna(
    subset=[
        "review",
        "result"
    ]
)


# ============================================================
# Clean text
# ============================================================

print()
print("Cleaning text...")

data["clean_review"] = data["review"].apply(
    clean_function
)


print(
    "Text cleaning completed."
)


# ============================================================
# Separate features and target
# ============================================================

X = data["clean_review"]

y = data["result"]


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# ============================================================
# TF-IDF
# ============================================================

print()
print("Creating TF-IDF vectors...")

tv = TfidfVectorizer()


X_train_vector = tv.fit_transform(
    X_train
)


X_test_vector = tv.transform(
    X_test
)


print(
    "TF-IDF completed."
)


# ============================================================
# Train Multinomial Naive Bayes
# ============================================================

print()
print(
    "Training Multinomial Naive Bayes..."
)


model = MultinomialNB()


model.fit(
    X_train_vector,
    y_train
)


print(
    "Model training completed."
)


# ============================================================
# Evaluate model
# ============================================================

predictions = model.predict(
    X_test_vector
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("=" * 60)

print(
    "MODEL ACCURACY:",
    accuracy
)

print("=" * 60)


print()
print(
    "Classification Report:"
)

print(
    classification_report(
        y_test,
        predictions
    )
)


# ============================================================
# Save model
# ============================================================

print()
print("Saving model.pkl...")

with open(
    "model.pkl",
    "wb"
) as f:

    dump(
        model,
        f
    )


# ============================================================
# Save TF-IDF vectorizer
# ============================================================

print(
    "Saving vector.pkl..."
)

with open(
    "vector.pkl",
    "wb"
) as f:

    dump(
        tv,
        f
    )


# ============================================================
# Finished
# ============================================================

print()
print("=" * 60)

print(
    "TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "Created:"
)

print(
    "1. model.pkl"
)

print(
    "2. vector.pkl"
)

print("=" * 60)