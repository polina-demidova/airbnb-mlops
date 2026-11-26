import os
import pickle
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights"
]

CATEGORICAL_FEATURES = [
    "neighbourhood_cleansed",
    "property_type",
    "room_type"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    """Загрузка датасета из S3."""

    load_dotenv()

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        region_name=os.environ["S3_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )

    response = s3.get_object(
        Bucket=os.environ["S3_BUCKET"],
        Key=os.environ["S3_KEY"]
    )

    df = pd.read_csv(response["Body"])

    print(f"Dataset loaded: {df.shape}")

    return df


def prepare_data(df):
    """Очистка данных и разделение выборки."""

    df = df.drop_duplicates().copy()

    df = df[
        df["price"].notna()
        & np.isfinite(df["price"])
        & (df["price"] > 0)
    ].reset_index(drop=True)

    print(f"Dataset after cleaning: {df.shape}")

    X = df[FEATURES].copy()
    y = df["price"].copy()
    groups = df["listing_id"].copy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_ids = set(groups.iloc[train_idx])
    test_ids = set(groups.iloc[test_idx])

    assert train_ids.isdisjoint(test_ids)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    print(f"Train: {X_train.shape}")
    print(f"Test: {X_test.shape}")
    print("Listing ID overlap: 0")

    return X_train, X_test, y_train, y_test


def build_pipeline():
    """Создание sklearn Pipeline."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    regressor = TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        func=np.log1p,
        inverse_func=np.expm1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ]
    )

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Оценка качества модели."""

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel metrics:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    return y_pred


def save_model(pipeline):
    """Сохранение модели в Docker volume."""

    model_path = Path(
        os.getenv("MODEL_PATH", "/models/model.pkl")
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(model_path, "wb") as file:
        pickle.dump(pipeline, file)

    size_mb = model_path.stat().st_size / 1024**2

    print(f"\nModel saved: {model_path}")
    print(f"Model size: {size_mb:.2f} MB")

    return model_path


def main():
    print("Starting batch training...")

    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    pipeline = build_pipeline()

    print("\nTraining Random Forest...")

    pipeline.fit(X_train, y_train)

    print("Training completed successfully")

    y_pred = evaluate_model(pipeline, X_test, y_test)

    model_path = save_model(pipeline)

    with open(model_path, "rb") as file:
        loaded_pipeline = pickle.load(file)

    np.testing.assert_allclose(
        y_pred,
        loaded_pipeline.predict(X_test),
        rtol=1e-10,
        atol=1e-10
    )

    print("Model serialization verified")
    print("Batch training completed successfully")


if __name__ == "__main__":
    main()