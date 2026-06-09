from airflow import DAG

from airflow.operators.bash import BashOperator

from datetime import datetime


default_args = {

    'owner': 'airflow',

    'start_date': datetime(2025, 1, 1)

}


with DAG(

    dag_id='voyage_training_pipeline',

    default_args=default_args,

    schedule_interval='@daily',

    catchup=False

) as dag:

    preprocess_task = BashOperator(

        task_id='preprocess_data',

        bash_command='echo Preprocessing Data'

    )

    train_regression_task = BashOperator(

        task_id='train_regression_model',

        bash_command='echo Training Regression Model'

    )

    evaluate_regression_task = BashOperator(

        task_id='evaluate_regression_model',

        bash_command='echo Evaluating Regression Model'

    )

    train_classification_task = BashOperator(

        task_id='train_classification_model',

        bash_command='echo Training Classification Model'

    )

    preprocess_task >> train_regression_task >> evaluate_regression_task >> train_classification_task
