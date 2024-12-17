## 📌 Project Overview
The *Trip Duration Predictor* is designed to predict the duration of taxi rides by analyzing various features such as geographic coordinates, time of day, and other relevant factors. By leveraging machine learning techniques, particularly *Ridge regression*, the project examines historical trip data to uncover patterns, enabling accurate predictions of trip durations in future scenarios.

## 📋 Contents
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Cluster Feature Extraction](#cluster-feature-extraction)
- [Data Visualization](#data-visualization)
- [Data Transformation](#data-transformation)
- [Machine Learning Model](#machine-learning-model)


## 🧹 Data Cleaning
The first step involved addressing outliers to ensure data quality. Removing or handling these outliers helped improve the model’s accuracy and made the data more consistent.

## 🔧 Feature Engineering
The dataset included geographical data such as longitude and latitude. I created new features like Haversine distance, Manhattan distance, and direction, which significantly improved the model's predictive power.

## 🔗 Cluster Feature Extraction
I utilized the *MiniBatchKMeans* algorithm to cluster pickup and dropoff coordinates into 'n' groups. This clustering allowed me to add new features such as *pickup_cluster* and *dropoff_cluster* to the dataset.

## 📊 Data Visualization
I performed data visualization to explore categorical data and identify patterns and trends. For numeric data, I discovered that it followed a right-skewed distribution, which led me to apply a log transformation to better understand and analyze the data.

## 📏 Data Transformation
Using *ColumnTransformer, I applied **OneHotEncoder* for categorical features and *StandardScaler* for numeric features. Additionally, I created new time-based features from datetime columns, such as extracting the *day* and *hour* from timestamps.

## 🛠 Machine Learning Model
I developed a machine learning model using a *Pipeline* that combines:
- *ColumnTransformer* for preprocessing
- *PolynomialFeatures* (degree 2) for feature interaction
- *Ridge regression* with an alpha value of 50 to ensure optimal model performance

This pipeline enhances the model's generalization ability, providing better predictions and more stable results.

## Results
| Metric            | Training R²          | Training RMSE         | Validation R²        | Validation RMSE      | Test R²             | Test RMSE           |
|-------------------|----------------------|-----------------------|-----------------------|----------------------|---------------------|---------------------|
| *Values*        | 0.7871               | 0.3531                | 0.7312                | 0.4148               | 0.7353              | 0.4095              |
