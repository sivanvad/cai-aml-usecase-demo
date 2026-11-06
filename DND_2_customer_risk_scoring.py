"""Score the engineered AML customer features with a distributed Spark ML model."""

import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

DATABASE = os.getenv("AML_DATABASE", "aml_demo_db")
STRING_FEATURES = ["entity_type", "country_of_residence", "onboarding_channel"]
EXCLUDED_FEATURES = {"customer_id", "historic_risk", "label"}


def build_pipeline(features_df):
    """Build preprocessing and classification stages from the input table schema."""
    available_columns = set(features_df.columns)
    required_columns = EXCLUDED_FEATURES | set(STRING_FEATURES)
    missing_columns = required_columns - available_columns
    if missing_columns:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing_columns)}")

    numeric_features = [
        field.name
        for field in features_df.schema.fields
        if isinstance(field.dataType, NumericType) and field.name not in EXCLUDED_FEATURES
    ]
    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_index", handleInvalid="keep")
        for column in STRING_FEATURES
    ]
    encoders = [
        OneHotEncoder(
            inputCol=indexer.getOutputCol(),
            outputCol=f"{indexer.getOutputCol()}_vector",
            handleInvalid="keep",
        )
        for indexer in indexers
    ]
    feature_columns = numeric_features + [encoder.getOutputCol() for encoder in encoders]
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="keep",
    )
    classifier = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        probabilityCol="probability",
        numTrees=100,
        maxDepth=6,
        seed=42,
    )
    return Pipeline(stages=indexers + encoders + [assembler, classifier]), numeric_features


def build_risk_reasons():
    """Return a Spark expression containing present, interpretable risk indicators."""
    indicators = [
        F.when(F.col("max_nlp_risk_score") > 0, F.lit("adverse_media")),
        F.when(F.col("pep_status") == 1, F.lit("pep_status")),
        F.when(F.col("occupation_risk") >= 4, F.lit("high_occupation_risk")),
        F.when(F.col("country_of_residence").isin("PA", "KY", "BS", "CY", "VG"), F.lit("high_risk_country")),
        F.when(F.col("cash_intensive_business") == 1, F.lit("cash_intensive_business")),
        F.when(F.col("high_risk_wire_volume") >= 25000, F.lit("high_risk_wire_volume")),
        F.when(F.col("crypto_to_total_ratio") >= 0.25, F.lit("crypto_activity")),
        F.when(F.col("shared_ip_count") > 0, F.lit("shared_ip")),
    ]
    return F.when(
        F.concat_ws(" | ", *indicators) == "",
        F.lit("No listed risk indicators"),
    ).otherwise(F.concat_ws(" | ", *indicators))


def main():
    spark = (
        SparkSession.builder.appName("AML_Customer_Risk_Scoring")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .enableHiveSupport()
        .getOrCreate()
    )

    try:
        input_table = f"{DATABASE}.customer_features_base"
        output_table = f"{DATABASE}.customer_risk_scores"
        print(f"Loading engineered features from {input_table}...")
        features_df = spark.table(input_table)
        pipeline, numeric_features = build_pipeline(features_df)

        # historic_risk is a synthetic proxy label, not a production SAR outcome.
        training_df = features_df.withColumn(
            "label",
            F.when(F.col("historic_risk") == "High", F.lit(1.0)).otherwise(F.lit(0.0)),
        ).na.fill(0, subset=numeric_features)

        print("Training the Spark ML risk model from the demo proxy label...")
        model = pipeline.fit(training_df)

        scored_df = model.transform(features_df.na.fill(0, subset=numeric_features)).select(
            "customer_id",
            F.round(F.col("probability")[1] * 100, 2).alias("customer_risk_score"),
            build_risk_reasons().alias("top_risk_factors"),
        )
        scored_df = scored_df.withColumn(
            "risk_tier",
            F.when(F.col("customer_risk_score") >= 80, F.lit("High"))
            .when(F.col("customer_risk_score") >= 50, F.lit("Medium"))
            .otherwise(F.lit("Low")),
        )

        scored_df.write.mode("overwrite").saveAsTable(output_table)
        print(f"Scoring complete. Results saved to {output_table}.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()