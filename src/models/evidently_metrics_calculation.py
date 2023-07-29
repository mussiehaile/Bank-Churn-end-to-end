import datetime
import pandas as pd
import psycopg
import joblib
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
import warnings

# Suppress the FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
# Load the pre-trained model
with open('/home/mussie/Music/home projects/nice_one/seving-ml-model-using-fastapi-and-docker/model.pkl', 'rb') as f_in:
    model = joblib.load(f_in)

# Load the raw data from CSV
raw_data = pd.read_csv('/home/mussie/Music/home projects/nice_one/seving-ml-model-using-fastapi-and-docker/Bank Customer Churn Prediction.csv')

# Define the features and column mapping
num_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary']
cat_features = []
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

# Load the reference data
reference_data = pd.read_parquet('data/reference.parquet')

# Create the Evidently Report
report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

# Calculate metrics and store in PostgreSQL
def calculate_and_store_metrics():
    current_data = raw_data.copy()
    current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    # Store metrics in PostgreSQL
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as curr:
            curr.execute(
                "INSERT INTO dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) VALUES (%s, %s, %s, %s)",
                (datetime.datetime.now(), prediction_drift, num_drifted_columns, share_missing_values)
            )

if __name__ == '__main__':
    calculate_and_store_metrics()
