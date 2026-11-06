"""
2a_model_training.py
Cloudera AI (CAI) AML Pipeline - MLOps Training Phase
-------------------------------------------------------------------
This script extracts a safe training sample from the feature store, 
trains the CatBoost behavioral anomaly model, and logs the trained 
artifact into the MLflow Model Registry for downstream scoring.
"""

import os
import pandas as pd
import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
from pyspark.sql import SparkSession

DATABASE = os.getenv("AML_DATABASE", "aml_demo_db")
EXPERIMENT_NAME = os.getenv("AML_MLFLOW_EXPERIMENT", "AML_Behavioral_Scoring")
TRAINING_SAMPLE_SIZE = max(1, int(os.getenv("AML_CATBOOST_TRAINING_SAMPLE_SIZE", "100000")))
TRAINING_SAMPLE_FRACTION = min(
    1.0, max(0.01, float(os.getenv("AML_CATBOOST_TRAINING_SAMPLE_FRACTION", "1.0")))
)


def main():
    spark = (
        SparkSession.builder.appName("AML_Model_Training")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .enableHiveSupport()
        .getOrCreate()
    )

    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"Loading master feature table from {DATABASE}...")
        features_df = spark.table(f"{DATABASE}.customer_features_base")

        feature_cols = [
            "age_years", "account_age_months", "declared_annual_income", 
            "tx_volume_peer_deviation", "cash_ratio_peer_deviation", 
            "crypto_to_total_ratio", "shared_ip_count", "linked_customer_count"
        ]

        required_columns = set(feature_cols) | {"historic_risk"}
        missing_columns = required_columns - set(features_df.columns)
        if missing_columns:
            raise ValueError(f"Feature table is missing required columns: {sorted(missing_columns)}")

        print("Preparing a bounded training set for CatBoost...")
        train_df = (
            features_df.select(*feature_cols, "historic_risk")
            .na.fill(0, subset=feature_cols)
            .sample(False, TRAINING_SAMPLE_FRACTION, seed=42)
            .limit(TRAINING_SAMPLE_SIZE)
            .toPandas()
        )
        
        if train_df.empty:
            raise ValueError("No rows available to train the behavioral model")

        X = train_df[feature_cols]
        y = (train_df["historic_risk"] == "High").astype("int8")
        
        if y.nunique() < 2:
            raise ValueError("Training sample must contain both High and non-High labels.")

        print("Training CatBoost Behavioral Engine...")
        with mlflow.start_run(run_name="CatBoost_Regular_Scoring"):
            model = CatBoostClassifier(
                iterations=150,
                learning_rate=0.1,
                depth=5,
                verbose=False,
                random_seed=42
            )
            model.fit(X, y)
            
            # Log model to MLflow for the scoring pipeline to retrieve
            mlflow.catboost.log_model(model, artifact_path="catboost_model")
            
            # Log parameters and metrics for model governance
            mlflow.log_params(model.get_params())
            mlflow.log_metric("training_samples", len(X))
            mlflow.log_param("training_sample_fraction", TRAINING_SAMPLE_FRACTION)
            mlflow.log_param("training_sample_limit", TRAINING_SAMPLE_SIZE)
            
            run_id = mlflow.active_run().info.run_id
            print(f"SUCCESS: Model trained and logged to MLflow (Run ID: {run_id}).")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()