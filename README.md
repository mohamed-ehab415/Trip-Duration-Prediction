# NYC Taxi Trip Duration Prediction

This project predicts the duration of taxi trips in New York City. The model uses Ridge Regression to predict trip durations. Key features include outlier handling, feature engineering (e.g., distance and direction), and data preprocessing. The code evaluates the model’s performance using RMSE and R-squared metrics.

## Key Features
- Outlier detection and data cleaning
- Feature engineering for time and spatial features
- MiniBatchKMeans clustering for spatial features
- Ridge Regression with PolynomialFeatures for prediction
- Model evaluation metrics: RMSE and R-squared

## Usage
1. Place your dataset in the `data/` directory.
2. Adjust file paths in `main.py` as needed.
3. Run `main.py` to train and evaluate the model.

## Requirements
- numpy
- pandas
- scikit-learn
- matplotlib
- pickle

## License
This project is licensed under the MIT License.

