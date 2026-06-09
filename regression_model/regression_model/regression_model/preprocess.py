import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data():

    flights = pd.read_csv('data/raw/flights.csv')

    return flights


def preprocess_data():

    flights = load_data()

    # Remove duplicates
    flights.drop_duplicates(inplace=True)

    # Handle missing values
    flights.fillna(method='ffill', inplace=True)

    # Convert date
    flights['date'] = pd.to_datetime(flights['date'])

    # Extract features
    flights['year'] = flights['date'].dt.year
    flights['month'] = flights['date'].dt.month
    flights['day'] = flights['date'].dt.day

    # Label Encoding
    le_from = LabelEncoder()
    le_to = LabelEncoder()
    le_agency = LabelEncoder()
    le_flight = LabelEncoder()

    flights['from'] = le_from.fit_transform(flights['from'])
    flights['to'] = le_to.fit_transform(flights['to'])
    flights['agency'] = le_agency.fit_transform(flights['agency'])
    flights['flightType'] = le_flight.fit_transform(
        flights['flightType']
    )

    # Features
    X = flights[
        [
            'from',
            'to',
            'distance',
            'time',
            'agency',
            'flightType',
            'year',
            'month',
            'day'
        ]
    ]

    # Target
    y = flights['price']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test
