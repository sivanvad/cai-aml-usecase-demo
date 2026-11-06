"""
1_feature_engineering.py
Cloudera AI (CAI) AML Pipeline - Stage 1 (Batch Feature Gathering)
-------------------------------------------------------------------
This script builds a highly scalable AML feature set entirely using 
native PySpark SQL functions. 

Key Optimization (Step 1 of the Pivot):
Introduces PySpark Window functions to calculate peer-group deviations. 
Instead of raw transaction volume, the behavioral ML engine receives 
the customer's z-score (deviation) relative to their specific industry segment.
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

DATABASE = os.getenv("AML_DATABASE", "aml_demo_db")


def build_transaction_features(transactions_df):
    """Calculates 30-day transaction velocities, volumes, and typologies."""
    tx_type = F.col("transaction_type")
    
    return transactions_df.groupBy("customer_id").agg(
        F.count("tx_id").alias("tx_30d_count"),
        F.sum("amount").alias("tx_30d_volume"),
        F.avg("amount").alias("tx_30d_avg_size"),
        F.sum(F.when(tx_type == "Crypto Purchase", F.col("amount")).otherwise(0)).alias("crypto_volume"),
        F.sum(F.when(tx_type == "International Wire", F.col("amount")).otherwise(0)).alias("high_risk_wire_volume"),
        F.sum(F.when(F.col("is_cross_border") == 1, F.col("amount")).otherwise(0)).alias("cross_border_volume"),
        F.sum(F.when(tx_type.isin("Cash Deposit", "Cash Withdrawal"), F.col("amount")).otherwise(0)).alias("cash_volume"),
    ).withColumn(
        "crypto_to_total_ratio", 
        F.col("crypto_volume") / (F.col("tx_30d_volume") + 1e-5)
    ).withColumn(
        "cash_to_total_ratio", 
        F.col("cash_volume") / (F.col("tx_30d_volume") + 1e-5)
    )


def add_peer_group_deviations(tx_features_df, customers_df):
    """
    Calculates peer-group deviations (z-scores) using native PySpark Window functions.
    This provides the behavioral ML engine with context: Is $50k in cash unusual 
    for a Retail Business, compared to an Individual in Professional Services?
    """
    # Join necessary demographics for partitioning
    base_df = tx_features_df.join(
        customers_df.select("customer_id", "entity_type", "industry_segment"),
        on="customer_id",
        how="inner"
    )

    # Define the peer group window
    peer_window = Window.partitionBy("entity_type", "industry_segment")

    # Calculate peer mean and standard deviation natively
    df_with_stats = base_df.withColumn(
        "peer_avg_tx_volume", F.avg("tx_30d_volume").over(peer_window)
    ).withColumn(
        "peer_std_tx_volume", F.stddev("tx_30d_volume").over(peer_window)
    ).withColumn(
        "peer_avg_cash_ratio", F.avg("cash_to_total_ratio").over(peer_window)
    )

    # Calculate deviations (z-scores)
    # Using coalesce to handle standard deviations of 0 (where all peers have exactly the same volume)
    deviations_df = df_with_stats.withColumn(
        "tx_volume_peer_deviation",
        F.when(F.col("peer_std_tx_volume") > 0, 
               (F.col("tx_30d_volume") - F.col("peer_avg_tx_volume")) / F.col("peer_std_tx_volume")
        ).otherwise(0.0)
    ).withColumn(
        "cash_ratio_peer_deviation",
        F.col("cash_to_total_ratio") - F.col("peer_avg_cash_ratio")
    )

    # Drop intermediate statistical columns and return feature set
    return deviations_df.drop(
        "entity_type", "industry_segment", "peer_avg_tx_volume", 
        "peer_std_tx_volume", "peer_avg_cash_ratio"
    )


def build_network_features(network_logs_df):
    """Compute shared-IP metrics without materializing a raw customer-pair join."""
    memberships = (
        network_logs_df.select("customer_id", "ip_address")
        .where(F.col("customer_id").isNotNull() & F.col("ip_address").isNotNull())
        .dropDuplicates()
    )
    shared_ip_groups = (
        memberships.groupBy("ip_address")
        .agg(F.collect_set("customer_id").alias("ip_customers"))
        .where(F.size("ip_customers") > 1)
    )

    shared_ip_counts = (
        memberships.join(shared_ip_groups.select("ip_address"), on="ip_address", how="inner")
        .groupBy("customer_id")
        .agg(F.count("ip_address").alias("shared_ip_count"))
    )
    shared_ip_usage = (
        network_logs_df.join(shared_ip_groups.select("ip_address"), on="ip_address", how="inner")
        .groupBy("customer_id")
        .agg(F.count("*").alias("shared_ip_usage_count"))
    )
    linked_customers = (
        memberships.join(shared_ip_groups, on="ip_address", how="inner")
        .groupBy("customer_id")
        .agg(
            F.array_distinct(F.flatten(F.collect_list("ip_customers"))).alias("linked_customer_ids")
        )
        .withColumn(
            "linked_customer_count",
            F.size(F.array_except("linked_customer_ids", F.array(F.col("customer_id")))),
        )
        .select("customer_id", "linked_customer_count")
    )

    return (
        shared_ip_counts.join(shared_ip_usage, on="customer_id", how="left")
        .join(linked_customers, on="customer_id", how="left")
    )


def build_media_features(adverse_media_df):
    """High-speed L0 triage on media using keyword heuristics in Spark SQL."""
    risk_keywords = [
        "money laundering", "fraud", "corruption", "terrorism financing",
        "sanctions evasion", "shell company", "ponzi", "embezzlement",
        "human trafficking", "tax evasion", "illicit funds", "wire fraud"
    ]

    article_text = F.lower(F.col("article_snippet"))
    
    risk_score_expr = sum(
        F.when(article_text.contains(keyword), 1.0).otherwise(0.0)
        for keyword in risk_keywords
    )

    scored_media = adverse_media_df.withColumn(
        "adverse_media_risk_score", 
        risk_score_expr / len(risk_keywords)
    )

    return scored_media.groupBy("customer_id").agg(
        F.max("adverse_media_risk_score").alias("max_nlp_risk_score"),
        F.count("*").alias("adverse_media_count"),
    )


def main():
    spark = (
        SparkSession.builder.appName("AML_Batch_Feature_Engineering")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .enableHiveSupport()
        .getOrCreate()
    )

    try:
        print(f"Loading raw tables from {DATABASE}...")
        customers_df = spark.table(f"{DATABASE}.customers")
        transactions_df = spark.table(f"{DATABASE}.transactions")
        network_logs_df = spark.table(f"{DATABASE}.network_logs")
        adverse_media_df = spark.table(f"{DATABASE}.adverse_media")

        print("1. Aggregating transaction behaviors...")
        tx_features_base = build_transaction_features(transactions_df)

        print("2. Calculating peer-group deviations via Spark Windows...")
        tx_features = add_peer_group_deviations(tx_features_base, customers_df)

        print("3. Computing distributed network risk features...")
        graph_features_df = build_network_features(network_logs_df)

        print("4. Scoring adverse media risk via L0 keyword triage...")
        nlp_features = build_media_features(adverse_media_df)

        print("5. Assembling the master base feature set...")
        master_features_df = (
            customers_df.select(
                "customer_id", "entity_type", "industry_segment", "age_years", 
                "country_of_residence", "account_age_months", "onboarding_channel", 
                "declared_annual_income", "cash_intensive_business", "pep_status", 
                "sanctions_hit", "occupation_risk", "historic_risk"
            )
            .join(tx_features, on="customer_id", how="left")
            .join(graph_features_df, on="customer_id", how="left")
            .join(nlp_features, on="customer_id", how="left")
            .fillna(0.0) # Zero out nulls for XGBoost/CatBoost input
        )

        output_table = f"{DATABASE}.customer_features_base"
        
        print(f"6. Writing enriched features to {output_table}...")
        master_features_df.write.mode("overwrite").saveAsTable(output_table)
        print("SUCCESS: Stage 1 Feature Engineering complete.")
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()