import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

print("Dataset loaded")
print(data.head())

# Keep only first two columns (label & message)
data = data.iloc[:, :2]
data.columns = ["label", "message"]

print("Columns fixed:", data.columns.tolist())

# Convert labels to numbers
data["label"] = data["label"].map({"spam": 1, "ham": 0})

print("Label conversion done")

# Features and target
X = data["message"]
y = data["label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

print("Text vectorization done")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained")

# Accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# ---- DEMO PREDICTION ----
sample_message = ["Congratulations! You won a free prize"]
sample_vector = vectorizer.transform(sample_message)
prediction = model.predict(sample_vector)

if prediction[0] == 1:
    print("Prediction for sample message: SPAM")
else:
    print("Prediction for sample message: NOT SPAM")
