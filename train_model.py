import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------
# 1. Load dataset
# --------------------------------
data = pd.read_csv("food_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# --------------------------------
# 2. Convert target labels to numbers
# --------------------------------
encoder = LabelEncoder()

data["Freshness"] = encoder.fit_transform(data["Freshness"])

# --------------------------------
# 3. Select input features
# --------------------------------
X = data[
    [
        "Temp_C",
        "Storage_Days",
        "Humidity",
        "Odor",
        "Color_Change",
        "Texture_Change"
    ]
]

y = data["Freshness"]

# --------------------------------
# 4. Split dataset
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------
# 5. Create ML model
# --------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# --------------------------------
# 6. Train model
# --------------------------------
model.fit(X_train, y_train)

# --------------------------------
# 7. Test model
# --------------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

# --------------------------------
# 8. Save model + encoder
# --------------------------------
joblib.dump(model, "food_freshness_model.pkl")
joblib.dump(encoder, "freshness_encoder.pkl")

print("\nModel saved as:")
print("food_freshness_model.pkl")

print("Encoder saved as:")
print("freshness_encoder.pkl")