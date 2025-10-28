
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):

    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    
    df = pd.read_parquet(url)

    # Calculate trip duration in minutes

    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60) 

    # Keep only trips that at least 1 minute and at most 60

    df = df[(df['duration'] >= 1) & (df['duration'] <= 60)]

    # Reduce dataset for perfomance. The model won't require so many data points

    df_sample = df.sample(frac=0.1, random_state=2)
    df_sample.shape

    # Features for modeling

    categorical = ['PULocationID', 'DOLocationID']

    df_sample[categorical] = df_sample[categorical].astype(str)
    
    df_sample['PU_DO'] = df_sample['PULocationID'] + "_" + df_sample['DOLocationID']
    
    return df_sample

def create_X(df, dv=None, ):

    categorical = ['PU_DO'] 
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_moodel(X_train, y_train, X_val, y_val, dv):

    with mlflow.start_run() as run:

        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
        "max_depth": 54,
        "learning_rate": 0.838560298111564,
        "reg_alpha": 0.07217623016548881,
        "reg_lambda": 0.0038449610359954913,
        "min_child_weight": 9.039435151170704,
        "objective": "reg:squarederror",
        "seed": 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
        params=best_params,
        dtrain=train,
        num_boost_round=30,
        evals=[(valid, "validation")],
        early_stopping_rounds=50)

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, name="models_mlflow")

        return run.info.run_id

def run(year, month):
    df_train =  read_dataframe(year=year, month=month)

    next_year = year if month > 12 else year + 1
    next_month = month + 1 if month > 12 else 1

    df_val =  read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_moodel(X_train, y_train, X_val, y_val, dv)

    return run_id

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Train a model to predict taxi duration trip")
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train the model')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train the model')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("run_id.txt", "w") as f:
        f.write(run_id)