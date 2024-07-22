"""

 "Most of the decisions made here in the code were a result of the exploratory data analysis (EDA).

"""


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)




def handle_outlier_and_clean_up(train):
     # Filter out invalid location coordinates
    train = train[(train['pickup_longitude'] <= -73.75) & (train['pickup_longitude'] >= -74.03)]
    train = train[(train['pickup_latitude'] <= 40.85) & (train['pickup_latitude'] >= 40.63)]
    train = train[(train['dropoff_longitude'] <= -73.75) & (train['dropoff_longitude'] >= -74.03)]
    train = train[(train['dropoff_latitude'] <= 40.85) & (train['dropoff_latitude'] >= 40.63)]

    #target
    m = np.mean(train['trip_duration'])
    s = np.std(train['trip_duration'])
    train = train[(train['trip_duration'] <= m + 2 * s) & (train['trip_duration'] >= m - 2 * s)]

    return train

def cluster_features(train, test, n_clusters=13, sample_size=500000, batch_size=10000, random_state=42):
    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values))
    sample_ind = np.random.permutation(len(coords))[:sample_size]
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    kmeans.fit(coords[sample_ind])
    train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
    train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
    test['pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
    test['dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
    return train, test, kmeans





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

def predict_eval(model, data, train_features, name):
    y_pred = model.predict(data[train_features])
    rmse = mean_squared_error(data.log_trip_duration, y_pred, squared=False)
    r2 = r2_score(data.log_trip_duration, y_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")

def approach1(train, test):
    numeric_features = ['distance_haversine_log', 'distance_dummy_manhattan_log', 'direction_sin','direction_cos']

    categorical_features = ['month','hour','dayofweek','weekdayhour','DayofMonth' ,'passenger_count',
                            'vendor_id', 'pickup_cluster', 'dropoff_cluster']
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features) #tha data is not normal distribution
    ], remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
        ('regression', Ridge(alpha=100,random_state=RANDOM_STATE))
    ])
    model = pipeline.fit(train[train_features], train.log_trip_duration)
    
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")

if __name__ == '__main__':
    RANDOM_STATE = 43
    np.random.seed(RANDOM_STATE)

    train = pd.read_csv(r"C:\Users\lap shop\Desktop\machin\projects\1 project-nyc-taxi-trip-duration\split\train.csv")
    test = pd.read_csv(r"C:\Users\lap shop\Desktop\machin\projects\1 project-nyc-taxi-trip-duration\split\val.csv")

    train = handle_outlier_and_clean_up(train)

    train, test, kmeans_model = cluster_features(train, test, n_clusters=13)
    train = prepare_data(train)
    test = prepare_data(test)

    approach1(train, test)


    # Save the k-means model

""""


"""

