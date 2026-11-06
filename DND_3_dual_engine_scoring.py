"""
2_dual_engine_scoring.py
Cloudera AI (CAI) AML Pipeline - Stage 2 & 3 (Scoring & Thresholding)
-------------------------------------------------------------------
Executes a dual-engine risk scoring architecture:
1. Rules Matrix: Deterministic triggers for hard regulatory lines (Sanctions, PEPs).
2. ML Engine: CatBoost classifier to detect behavioral anomalies using peer deviations.

Final scores are blended using a MAX() function. A high-recall threshold 
is applied to push alerts into an Enhanced Due Diligence (EDD) queue for human/agent review.
"""

import os
import pandas as pd
from catboost import CatBoostClassifier
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

DATABASE = os.getenv("AML_DATABASE", "aml_demo_db")
EDD_THRESHOLD = 70.0  # High-recall threshold to cast a wider net
TRAINING_SAMPLE_SIZE = max(1, int(os.getenv("AML_CATBOOST_TRAINING_SAMPLE_SIZE", "100000")))
TRAINING_SAMPLE_FRACTION = min(
    1.0,
    max(0.01, float(os.getenv("AML_CATBOOST_TRAINING_SAMPLE_FRACTION", "0.2"))),
)


def apply_rules_engine(df):
    """
    Executes the deterministic Rules Matrix natively in Spark SQL.
    Returns a 'rule_score' from 0 to 100 based on hard regulatory red lines.
    """
    return df.withColumn(
        "rule_score",
        F.when(F.col("sanctions_hit") == 1, 100.0)
        .when(F.col("pep_status") == 2, 95.0)  # Foreign/Hard PEP
        .when(F.col("max_nlp_risk_score") > 0.85, 90.0)  # Severe adverse media
        .when(F.col("pep_status") == 1, 75.0)  # Domestic PEP
        .otherwise(0.0)
    ).withColumn(
        "rule_reason",
        F.when(F.col("sanctions_hit") == 1, "Sanctions Match")
        .when(F.col("pep_status") == 2, "Foreign PEP")
        .when(F.col("max_nlp_risk_score") > 0.85, "Severe Adverse Media")
        .when(F.col("pep_status") == 1, "Domestic PEP")
        .otherwise("None")
    )


def train_behavioral_model(features_df, feature_cols):
    """
    Trains a CatBoost model to detect behavioral anomalies.
    In a production MLOps pipeline, this would fetch a pre-trained model from MLflow.
    For this workflow, we train dynamically using historic_risk as the target.
    """
    required_columns = set(feature_cols) | {"historic_risk"}
    missing_columns = required_columns - set(features_df.columns)
    if missing_columns:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing_columns)}")

    print("Sampling training data for CatBoost...")
    train_df = (
        features_df.sample(False, TRAINING_SAMPLE_FRACTION, seed=42)
        .select(*feature_cols, "historic_risk")
        .limit(TRAINING_SAMPLE_SIZE)
        .toPandas()
    )
    if train_df.empty:
        raise ValueError("No rows available to train the behavioral model")

    X = train_df[feature_cols].fillna(0)
    y = (train_df["historic_risk"] == "High").astype("int8")
    if y.nunique() < 2:
        raise ValueError("Training sample must contain both High and non-High historic_risk labels")

    print("Training CatBoost Behavioral Engine...")
    model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.1,
        depth=5,
        verbose=False,
        random_seed=42
    )
    model.fit(X, y)
    return model


def main():
    spark = (
        SparkSession.builder.appName("AML_Dual_Engine_Scoring")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
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
        feature_cols = [
            "age_years", "account_age_months", "declared_annual_income", 
            "tx_volume_peer_deviation", "cash_ratio_peer_deviation", 
            "crypto_to_total_ratio", "shared_ip_count", "linked_customer_count"
        ]
        
        catboost_model = train_behavioral_model(features_df, feature_cols)
        
        # Broadcast the trained model to worker nodes to prevent serialization overhead
        broadcast_model = spark.sparkContext.broadcast(catboost_model)

        # Define a Pandas UDF for distributed inference
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
            model_features = pd.concat(
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
                axis=1,
            ).fillna(0)
            model_features.columns = feature_cols
            scores = broadcast_model.value.predict_proba(model_features)[:, 1] * 100.0
            return pd.Series(scores)

        print("Executing distributed ML inference...")
        # Apply the UDF to generate ML scores
        scored_df = ruled_df.withColumn(
            "ml_score", 
            predict_ml_score(*[F.col(column) for column in feature_cols])
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

        # Select output columns for clarity
        output_cols = [
            "customer_id", "entity_type", "industry_segment", 
            "rule_score", "rule_reason", "ml_score", "final_risk_score", "routing_decision"
        ]
        final_scores_df = final_df.select(output_cols)

        # 4. Write back to Cloudera Data Warehouse
        scores_table = f"{DATABASE}.customer_risk_scores"
        print(f"Writing all scores to {scores_table}...")
        final_scores_df.write.mode("overwrite").saveAsTable(scores_table)

        # Create the specific EDD Queue table for the AI Agents / Investigators to pick up
        edd_queue_table = f"{DATABASE}.edd_alerts_queue"
        print(f"Routing high-recall alerts to {edd_queue_table}...")
        edd_df = spark.table(scores_table).filter(F.col("routing_decision") == "EDD_Queue")
        edd_df.write.mode("overwrite").saveAsTable(edd_queue_table)

        print("SUCCESS: Scoring complete. EDD alerts routed to the queue table.")

    finally:
        spark.stop()

if __name__ == "__main__":
    main()