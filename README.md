🚗 1. Car Price Predictor

Overview

Car Price Predictor is a machine learning application designed to estimate the market price of Ford vehicles based on their specifications, usage history, and other relevant attributes. The project explores how different vehicle characteristics influence pricing and applies regression techniques to generate data-driven price predictions.

Features

Predicts vehicle prices using multiple input features.

Processes categorical variables using One-Hot Encoding.

Standardizes numerical features using StandardScaler.

Uses a Linear Regression model for price estimation.

Evaluates model performance through multiple regression metrics.

Methodology

The dataset is processed by separating the target variable from the independent features. Transmission, fuel type, and car model are encoded into numerical representations, while numerical features are standardized. The dataset is split into training and testing sets, and a Linear Regression model is trained and evaluated using R², Adjusted R², Mean Squared Error, and Root Mean Squared Error.

Tech Stack: Python, Pandas, NumPy, Scikit-learn, Streamlit

Live Demo: https://car-price-predictor-1-wib4.onrender.com

❤️ 2. Heart Disease Predictor

Overview

Heart Disease Predictor is a machine learning classification application designed to analyze health-related input features and predict the likelihood of heart disease. It explores multiple classification algorithms to compare their predictive performance and identify the most effective model for the dataset.

Features

Implements five different classification algorithms.

Performs dataset inspection and missing-value analysis.

Identifies correlations between input variables and the target.

Removes duplicate records.

Uses stratified train-test splitting.

Standardizes input features.

Compares models using Accuracy and F1 Score.

Visualizes model performance.

Saves trained models and preprocessing components for deployment.

Methodology

The dataset is inspected for missing values, correlations, and duplicates before being separated into features and target. The data is divided using stratified sampling and standardized. Logistic Regression, KNN, Naive Bayes, Decision Tree, and SVM are trained and evaluated using classification metrics. Results are compared to identify the best-performing model, which is serialized alongside the scaler and feature-column information.

Tech Stack: Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Joblib, Streamlit

Live Demo: https://heart-disease-predictor-tbwnk4mk4jrb2tkpxuj9h8.streamlit.app/
