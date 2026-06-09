import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocess import preprocess_data


# Load data
X_train, X_test, y_train, y_test = preprocess_data()

mlflow.set_tracking_uri(
    'file:../mlruns'
)


# SET EXPERIMENT NAME
mlflow.set_experiment(
    'Flight_Price_Regression'
)


# START RUN
with mlflow.start_run():

    print('MLflow Started')

    # Parameters
    n_estimators = 100
    random_state = 42

    # Create model
    model = RandomForestRegressor(

        n_estimators=n_estimators,
        random_state=random_state

    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    # Print metrics
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2 Score:', r2)

    # Log parameters
    mlflow.log_param(
        'n_estimators',
        n_estimators
    )

    mlflow.log_param(
        'random_state',
        random_state
    )

    # Log metrics
    mlflow.log_metric('MAE', mae)

    mlflow.log_metric('MSE', mse)

    mlflow.log_metric('R2_Score', r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
        'flight_price_model'
    )

    # Save model
    joblib.dump(

        model,

        'models/flight_price_model.pkl'

    )

    print('Regression Model Trained & Logged')
