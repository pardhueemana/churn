# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow import keras


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)


# =========================
# DATA CLEANING
# =========================

# Remove customerID
df.drop('customerID', axis=1, inplace=True)

# Handle TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Fill missing values
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

print(df.isnull().sum())


# =========================
# YES/NO ENCODING
# =========================

yes_no_columns = [
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'Churn'
]

for col in yes_no_columns:
    df[col] = df[col].replace({
        'Yes': 1,
        'No': 0
    })


# =========================
# ONE HOT ENCODING
# =========================

df = pd.get_dummies(
    df,
    columns=[
        'gender',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaymentMethod'
    ],
    drop_first=True
)

# Convert True/False to 1/0
for col in df.columns:
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)

print(df.head())


# =========================
# FEATURE SCALING
# =========================

cols_to_scale = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

scaler = MinMaxScaler()

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


# =========================
# SPLIT FEATURES & TARGET
# =========================

X = df.drop('Churn', axis=1)
y = df['Churn']
print(len(X.columns))
print(X.columns.tolist())
print(X.shape)
print(y.shape)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)


# =========================
# BUILD ANN MODEL
# =========================

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(16, activation='relu'),

    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =========================
# TRAIN MODEL
# =========================

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)


# =========================
# EVALUATE MODEL
# =========================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")


# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5).astype(int)

print(classification_report(
    y_test,
    y_pred
))


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# =========================
# SAVE MODEL
# =========================

model.save("customer_churn_model.keras")

print("Model saved successfully!")
import joblib

joblib.dump(X.columns.tolist(), "model_columns.pkl")
import os

print(os.listdir())
import joblib

cols = joblib.load("model_columns.pkl")

print(len(cols))
print(cols)