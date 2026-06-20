# 📊 Customer Churn Prediction using Artificial Neural Network (ANN)

## 🚀 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. This project uses an Artificial Neural Network (ANN) built with TensorFlow and Keras to predict whether a customer is likely to leave the service based on demographic and service-related information.

The model is trained on the Telco Customer Churn dataset and deployed using Streamlit for real-time predictions.

---

## 🎯 Objectives

* Analyze customer behavior and service usage.
* Predict customer churn using Machine Learning and Deep Learning.
* Build an interactive web application for real-time predictions.
* Help businesses identify high-risk customers and improve retention strategies.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Matplotlib
* Seaborn
* Joblib

---

## 📂 Dataset

Dataset: Telco Customer Churn Dataset

Features include:

* Gender
* Senior Citizen Status
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Online Security
* Online Backup
* Device Protection
* Tech Support
* Streaming TV
* Streaming Movies
* Contract Type
* Payment Method
* Monthly Charges
* Total Charges

Target Variable:

* Churn (Yes / No)

---

## 🔄 Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns (customerID)
* Handled missing values in TotalCharges
* Converted categorical variables into numerical values
* Applied One-Hot Encoding
* Applied MinMax Feature Scaling
* Split data into training and testing datasets

---

## 🧠 Model Architecture

Artificial Neural Network (ANN)

Input Layer: 30 Features

Hidden Layers:

* Dense Layer (64 Neurons, ReLU)
* Dropout Layer (0.3)
* Dense Layer (32 Neurons, ReLU)
* Dropout Layer (0.2)
* Dense Layer (16 Neurons, ReLU)

Output Layer:

* Dense Layer (1 Neuron, Sigmoid)

Loss Function:

* Binary Crossentropy

Optimizer:

* Adam

---

## 📈 Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 79.13% |
| Precision | 83%    |
| Recall    | 90%    |
| F1 Score  | 86%    |

The model achieves reliable performance in identifying customers likely to churn.

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── churn.ipynb
├── customer_churn_model.keras
├── scaler.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## ▶️ Running the Project Locally

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Customer-Churn-Prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🌐 Application Features

* Interactive user interface
* Real-time churn prediction
* Customer risk assessment
* Probability-based prediction output
* Easy deployment using Streamlit Cloud

---

## 📊 Future Improvements

* Implement XGBoost and Random Forest models
* Improve model accuracy through hyperparameter tuning
* Add customer segmentation dashboard
* Deploy using Docker and Cloud Services
* Integrate real-time database connectivity

---

## 👨‍💻 Author

**Pardhu Eemana**

AI & Machine Learning Enthusiast
