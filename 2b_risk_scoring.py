"""
2b_risk_scoring.py
Cloudera AI (CAI) AML Pipeline - Dual-Engine Scoring Phase
-------------------------------------------------------------------
This script executes the daily production scoring run. It applies the 
deterministic Rules Matrix natively in PySpark, fetches the latest 
trained ML model from MLflow, and uses a Pandas UDF to execute 
distributed inference. High-risk alerts are routed to the EDD queue.
"""

import os
import pandas as pd
import mlflow.catboost
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

DATABASE = os.getenv("AML_DATABASE", "aml_demo_db")
EXPERIMENT_NAME = os.getenv("AML_MLFLOW_EXPERIMENT", "AML_Behavioral_Scoring")
TRAINING_RUN_NAME = "CatBoost_Regular_Scoring"
EDD_THRESHOLD = 70.0
FEATURE_COLS = [
    "age_years",
    "account_age_months",
    "declared_annual_income",
    "tx_volume_peer_deviation",
    "cash_ratio_peer_deviation",
    "crypto_to_total_ratio",
    "shared_ip_count",
    "linked_customer_count",
]


def apply_rules_engine(df):
    """Executes the deterministic Rules Matrix natively in Spark SQL."""
    return df.withColumn(
        "rule_score",
        F.when(F.col("sanctions_hit") == 1, 100.0)
        .when(F.col("pep_status") == 2, 95.0)  
        .when(F.col("max_nlp_risk_score") > 0.85, 90.0)  
        .when(F.col("pep_status") == 1, 75.0)  
        .otherwise(0.0)
    ).withColumn(
        "rule_reason",
        F.when(F.col("sanctions_hit") == 1, "Sanctions Match")
        .when(F.col("pep_status") == 2, "Foreign PEP")
        .when(F.col("max_nlp_risk_score") > 0.85, "Severe Adverse Media")
        .when(F.col("pep_status") == 1, "Domestic PEP")
        .otherwise("None")
    )


def fetch_latest_model():
    """Load the latest completed model logged by the training workflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' was not found")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and tags.mlflow.runName = '{TRAINING_RUN_NAME}'"
        ),
        order_by=["start_time DESC"],
        max_results=1,
    )
    
    if runs.empty:
        raise RuntimeError(
            f"No completed '{TRAINING_RUN_NAME}' model found in MLflow. "
            "Run 2a_model_training.py first."
        )
        
    latest_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{latest_run_id}/catboost_model"
    print(f"Loading latest MLflow model from Run ID: {latest_run_id}")
    return mlflow.catboost.load_model(model_uri)


def main():
    spark = (
        SparkSession.builder.appName("AML_Dual_Engine_Scoring")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .config("spark.sql.adaptive.enabled", "true")
        .enableHiveSupport()
        .getOrCreate()
    )

    try:
        print(f"Loading master feature table from {DATABASE}...")
        features_df = spark.table(f"{DATABASE}.customer_features_base")

        # 1. Apply Deterministic Rules Matrix
        print("Executing Rules Matrix...")
        ruled_df = apply_rules_engine(features_df)

        # 2. Setup Behavioral ML Engine
        required_columns = set(FEATURE_COLS) | {
            "customer_id",
            "entity_type",
            "industry_segment",
            "sanctions_hit",
            "pep_status",
            "max_nlp_risk_score",
        }
        missing_columns = required_columns - set(features_df.columns)
        if missing_columns:
            raise ValueError(f"Feature table is missing required columns: {sorted(missing_columns)}")
        
        # Load model from MLflow and broadcast to worker nodes
        catboost_model = fetch_latest_model()
        broadcast_model = spark.sparkContext.broadcast(catboost_model)

        @F.pandas_udf(DoubleType())
        def predict_ml_score(
            age_years: pd.Series,
            account_age_months: pd.Series,
            declared_annual_income: pd.Series,
            tx_volume_peer_deviation: pd.Series,
            cash_ratio_peer_deviation: pd.Series,
            crypto_to_total_ratio: pd.Series,
            shared_ip_count: pd.Series,
            linked_customer_count: pd.Series,
        ) -> pd.Series:
            model_features = pd.DataFrame(
                dict(
                    zip(
                        FEATURE_COLS,
                        [
                            age_years,
                            account_age_months,
                            declared_annual_income,
                            tx_volume_peer_deviation,
                            cash_ratio_peer_deviation,
                            crypto_to_total_ratio,
                            shared_ip_count,
                            linked_customer_count,
                        ],
                    )
                )
            ).fillna(0)
            scores = broadcast_model.value.predict_proba(model_features)[:, 1] * 100.0
            return pd.Series(scores)

        print("Executing distributed ML inference via Pandas UDF...")
        scored_df = ruled_df.withColumn(
            "ml_score", 
            predict_ml_score(*[F.col(column) for column in FEATURE_COLS])
        )

        # 3. Blend Scores and Apply High-Recall Thresholding
        print("Blending scores and identifying EDD alerts...")
        final_df = scored_df.withColumn(
            "final_risk_score",
            F.greatest(F.col("rule_score"), F.col("ml_score"))
        ).withColumn(
            "routing_decision",
            F.when(F.col("final_risk_score") >= EDD_THRESHOLD, "EDD_Queue")
            .otherwise("Standard_Monitoring")
        )

        output_cols = [
            "customer_id", "entity_type", "industry_segment", 
            "rule_score", "rule_reason", "ml_score", "final_risk_score", "routing_decision"
        ]
        final_scores_df = final_df.select(output_cols)

        # 4. Write back to Cloudera Data Warehouse
        scores_table = f"{DATABASE}.customer_risk_scores"
        print(f"Writing all scores to {scores_table}...")
        final_scores_df.write.mode("overwrite").saveAsTable(scores_table)

        edd_queue_table = f"{DATABASE}.edd_alerts_queue"
        print(f"Routing high-recall alerts to {edd_queue_table}...")
        
        # Read from materialized table to avoid DAG double-execution
        edd_df = spark.table(scores_table).filter(F.col("routing_decision") == "EDD_Queue")
        edd_df.write.mode("overwrite").saveAsTable(edd_queue_table)

        print("SUCCESS: Production scoring complete. EDD queue is ready for Agentic Enrichment.")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()