import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import pandas as pd

def prepare_data(data):
    # for datatime
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['DayofMonth'] = data['pickup_datetime'].dt.day
    data['dayofweek'] = data['pickup_datetime'].dt.dayofweek
    data['month'] = data['pickup_datetime'].dt.month
    data['hour'] = data['pickup_datetime'].dt.hour
    data['dayofyear'] = data['pickup_datetime'].dt.dayofyear
    data['weekdayhour'] = data['hour'] * data['dayofweek']

# for distance and dirction
    def haversine_array(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371  # in km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
        a = haversine_array(lat1, lng1, lat1, lng2)
        b = haversine_array(lat1, lng1, lat2, lng1)
        return a + b

    def bearing_array(lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))
    data.loc[:, 'distance_haversine'] = haversine_array(data['pickup_latitude'].values, data['pickup_longitude'].values,
                                                    data['dropoff_latitude'].values, data['dropoff_longitude'].values)

    data.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(data['pickup_latitude'].values,
                                                                   data['pickup_longitude'].values, data['dropoff_latitude'].values, data['dropoff_longitude'].values)

    data.loc[:, 'direction'] = bearing_array(data['pickup_latitude'].values, data['pickup_longitude'].values,
                                             data['dropoff_latitude'].values, data['dropoff_longitude'].values)

    data['distance_haversine_log'] = np.log1p(data['distance_haversine'])
    data['distance_dummy_manhattan_log'] = np.log1p(data['distance_dummy_manhattan'])
    data['direction_sin'] = np.sin(np.radians(data['direction']))
    data['direction_cos'] = np.cos(np.radians(data['direction']))

#target

    data['log_trip_duration'] = np.log(data['trip_duration'])

    return data

def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")



if __name__ == '__main__':
        # import  model
        import pickle
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        #read data
        path=''
        test = pd.read_csv(path)

        # prepar data
        test = prepare_data(test)
        test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
        test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

        numeric_features = ['distance_haversine_log', 'distance_dummy_manhattan_log', 'direction_sin','direction_cos']

        categorical_features = ['month','hour','dayofweek','weekdayhour','DayofMonth' ,'passenger_count',
                            'vendor_id', 'pickup_cluster', 'dropoff_cluster']
        test_features = categorical_features + numeric_features

        # Evaluate the model on test data
        predict_eval(model, test, test_features, "test")

