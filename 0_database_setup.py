import os

from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F

STANDARD_COUNTRIES = ["US", "UK", "CA", "AU", "DE", "FR", "JP"]
HIGH_RISK_COUNTRIES = ["PA", "KY", "BS", "CY", "VG"]
TX_TYPES = [
    "Cash Deposit",
    "Cash Withdrawal",
    "Domestic Wire",
    "International Wire",
    "ACH Transfer",
    "P2P Payment",
    "Crypto Purchase",
]
TX_WEIGHTS = [0.15, 0.10, 0.15, 0.10, 0.30, 0.15, 0.05]
INDIVIDUAL_SEGMENTS = ["Professional Services", "Retail", "Unemployed"]
INDIVIDUAL_WEIGHTS = [0.6, 0.3, 0.1]
CORPORATE_SEGMENTS = ["Retail", "Real Estate", "Crypto/MSB", "Manufacturing"]
CORPORATE_WEIGHTS = [0.4, 0.3, 0.1, 0.2]
DATABASE = os.getenv("AML_DATABASE", "aml_demo_db")


def env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def weighted_choice(values, weights, seed):
    random_value = F.rand(seed)
    threshold = 0.0
    expression = None
    for value, weight in zip(values[:-1], weights[:-1]):
        threshold += weight
        choice = F.when(random_value < threshold, F.lit(value))
        expression = choice if expression is None else expression.when(random_value < threshold, F.lit(value))
    return expression.otherwise(F.lit(values[-1]))


def random_array_value(values, seed):
    index = (F.floor(F.rand(seed) * len(values)) + 1).cast("int")
    return F.element_at(F.array(*[F.lit(value) for value in values]), index)


def build_customer_df(spark, count, seed):
    customers = spark.range(1, count + 1).withColumn(
        "entity_type",
        weighted_choice(["Individual", "Corporate"], [0.85, 0.15], seed),
    )
    customers = customers.withColumn(
        "age_years",
        F.when(
            F.col("entity_type") == "Individual",
            F.floor(F.rand(seed + 1) * 68 + 18),
        ).otherwise(F.floor(F.rand(seed + 2) * 30 + 1)),
    )
    customers = customers.withColumn(
        "country_of_residence",
        F.when(F.rand(seed + 3) < 0.05, random_array_value(HIGH_RISK_COUNTRIES, seed + 4))
        .otherwise(random_array_value(STANDARD_COUNTRIES, seed + 5)),
    )
    customers = customers.withColumn(
        "onboarding_channel",
        weighted_choice(
            ["In-Branch", "Digital-KYC", "Third-Party-Broker"],
            [0.4, 0.5, 0.1],
            seed + 6,
        ),
    )
    customers = customers.withColumn(
        "industry_segment",
        F.when(
            F.col("entity_type") == "Individual",
            weighted_choice(INDIVIDUAL_SEGMENTS, INDIVIDUAL_WEIGHTS, seed + 7),
        ).otherwise(weighted_choice(CORPORATE_SEGMENTS, CORPORATE_WEIGHTS, seed + 8)),
    )
    customers = customers.withColumn(
        "declared_annual_income",
        F.round(
            F.when(
                F.col("entity_type") == "Individual",
                30000 + F.rand(seed + 9) * 220000,
            ).otherwise(100000 + F.rand(seed + 10) * 9900000),
            2,
        ),
    )
    customers = customers.withColumn(
        "cash_intensive_business",
        F.when((F.col("entity_type") == "Corporate") & (F.rand(seed + 11) < 0.3), 1).otherwise(0),
    ).withColumn("pep_status", weighted_choice([0, 1, 2], [0.95, 0.04, 0.01], seed + 12))
    customers = customers.withColumn(
        "sanctions_hit",
        F.when(F.rand(seed + 13) < 0.005, 1).otherwise(0),
    ).withColumn("account_age_months", F.floor(F.rand(seed + 14) * 120 + 1))
    customers = customers.withColumn(
        "occupation_risk",
        F.floor(F.rand(seed + 15) * 5 + 1),
    )

    risk_logit = (
        F.lit(-3.5)
        + F.col("sanctions_hit") * 4.0
        + F.when(F.col("pep_status") == 2, 2.6).otherwise(0.0)
        + F.when(F.col("pep_status") == 1, 0.8).otherwise(0.0)
        + F.when(F.col("country_of_residence").isin(HIGH_RISK_COUNTRIES), 1.0).otherwise(0.0)
        + (F.col("occupation_risk") - 1) * 0.22
        + F.col("cash_intensive_business") * 0.45
        + F.when(F.col("industry_segment") == "Crypto/MSB", 0.65).otherwise(0.0)
        + F.when(F.col("onboarding_channel") == "Third-Party-Broker", 0.25).otherwise(0.0)
    )
    latent_risk_propensity = 1.0 / (1.0 + F.exp(-risk_logit))
    medium_probability = (1.0 - latent_risk_propensity) * (0.35 + 0.25 * latent_risk_propensity)
    label_draw = F.rand(seed + 16)
    customers = customers.withColumn(
        "historic_risk",
        F.when(label_draw < latent_risk_propensity, "High")
        .when(label_draw < latent_risk_propensity + medium_probability, "Medium")
        .otherwise("Low"),
    )

    return customers.select(
        F.format_string("C%05d", "id").alias("customer_id"),
        "entity_type",
        "industry_segment",
        "age_years",
        "country_of_residence",
        "account_age_months",
        "onboarding_channel",
        "declared_annual_income",
        "cash_intensive_business",
        "pep_status",
        "sanctions_hit",
        "occupation_risk",
        "historic_risk",
    )


def build_transaction_df(spark, count, customer_count, seed):
    transactions = spark.range(1, count + 1).withColumn(
        "transaction_type",
        weighted_choice(TX_TYPES, TX_WEIGHTS, seed),
    )
    transaction_type = F.col("transaction_type")
    is_cash = transaction_type.isin("Cash Deposit", "Cash Withdrawal")
    amount = (
        F.when(transaction_type == "International Wire", 5000 + F.rand(seed + 1) * 145000)
        .when(is_cash, 100 + F.rand(seed + 2) * 9850)
        .when(transaction_type == "Crypto Purchase", 500 + F.rand(seed + 3) * 24500)
        .otherwise(50 + F.rand(seed + 4) * 74950)
    )
    cross_border = (
        F.when(transaction_type == "International Wire", 1)
        .when(is_cash, 0)
        .when(transaction_type == "Crypto Purchase", F.floor(F.rand(seed + 5) * 2))
        .otherwise(F.when(F.rand(seed + 6) > 0.9, 1).otherwise(0))
    )
    return transactions.select(
        F.format_string("TX%07d", "id").alias("tx_id"),
        F.format_string(
            "C%05d",
            F.floor(F.rand(seed + 7) * customer_count + 1).cast("int"),
        ).alias("customer_id"),
        F.round(amount, 2).alias("amount"),
        cross_border.alias("is_cross_border"),
        "transaction_type",
    )


def build_network_df(spark, count, customer_count, seed):
    logs = spark.range(1, count + 1)
    shared_ip = F.array(F.lit("192.168.1.50"), F.lit("10.0.0.99"))
    shared_index = (F.floor(F.rand(seed + 1) * 2) + 1).cast("int")
    random_ip = F.concat(F.lit("192.168.1."), (F.floor(F.rand(seed + 2) * 156) + 100).cast("string"))
    return logs.select(
        F.format_string(
            "C%05d",
            (F.pmod(F.col("id") - 1, F.lit(customer_count)) + 1).cast("int"),
        ).alias("customer_id"),
        F.when(F.rand(seed) > 0.95, F.element_at(shared_ip, shared_index))
        .otherwise(random_ip)
        .alias("ip_address"),
    )


def build_feedback_df(spark, count, customer_count, seed):
    feedback = spark.range(1, count + 1)
    return feedback.select(
        F.format_string("ALT%05d", "id").alias("alert_id"),
        F.format_string(
            "C%05d",
            (F.floor(F.rand(seed) * customer_count) + 1).cast("int"),
        ).alias("customer_id"),
        weighted_choice(["True Positive", "False Positive"], [0.15, 0.85], seed + 1)
        .alias("investigator_disposition"),
        F.date_format(
            F.date_sub(F.current_date(), F.floor(F.rand(seed + 2) * 91).cast("int")),
            "yyyy-MM-dd",
        ).alias("review_date"),
    )


def write_table(df, table_name, expected_count):
    table = f"{DATABASE}.{table_name}"
    df.write.mode("overwrite").saveAsTable(table)
    print(f"Created {table} with {expected_count} records.")


def remap_customer_ids(rows, customer_count):
    mapped_rows = []
    for row in rows:
        values = row.asDict()
        customer_number = int(values["customer_id"][1:])
        values["customer_id"] = f"C{(customer_number - 1) % customer_count + 1:05d}"
        mapped_rows.append(Row(**values))
    return mapped_rows


def main():
    customer_count = env_int("AML_CUSTOMER_COUNT", 5000)
    transaction_count = env_int("AML_TRANSACTION_COUNT", 15000)
    network_count = env_int("AML_NETWORK_COUNT", 5000)
    feedback_count = env_int("AML_FEEDBACK_COUNT", 500)
    seed = env_int("AML_RANDOM_SEED", 42)

    spark = SparkSession.builder.appName("AML_Database_Setup").enableHiveSupport().getOrCreate()
    try:
        spark.sql(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
        print(f"Database '{DATABASE}' is ready.")

        write_table(build_customer_df(spark, customer_count, seed), "customers", customer_count)
        write_table(
            build_transaction_df(spark, transaction_count, customer_count, seed + 100),
            "transactions",
            transaction_count,
        )
        write_table(
            build_network_df(spark, network_count, customer_count, seed + 200),
            "network_logs",
            network_count,
        )
        write_table(
            build_feedback_df(spark, feedback_count, customer_count, seed + 300),
            "compliance_feedback",
            feedback_count,
        )

        media_data = [
            Row(customer_id="C00420", article_snippet="Authorities are investigating the holding company for international money laundering and tax evasion."),
            Row(customer_id="C00111", article_snippet="CEO indicted on federal charges of orchestrating a $50 million cryptocurrency Ponzi scheme and wire fraud."),
            Row(customer_id="C00444", article_snippet="Leaked documents reveal the entity's direct ties to sanctioned oligarchs through a network of offshore shell companies."),
            Row(customer_id="C00666", article_snippet="Arrest warrant issued for the suspect linked to a cross-border human trafficking and illicit funds network."),
            Row(customer_id="C00888", article_snippet="Regulatory bodies are scrutinizing the trade finance firm over suspected invoice manipulation and phantom shipments."),
            Row(customer_id="C01234", article_snippet="Financial watchdog issues a warning about the unlicensed brokerage firm soliciting illicit deposits from retail investors."),
            Row(customer_id="C00222", article_snippet="The bank faces a $2M fine for failing to implement adequate KYC procedures, though no active money laundering was proven."),
            Row(customer_id="C00050", article_snippet="The individual was recently cleared of all allegations regarding corporate fraud and reinstated to the board."),
            Row(customer_id="C00333", article_snippet="The individual was named as a victim in a widespread phishing scam that targeted local retail businesses."),
            Row(customer_id="C00777", article_snippet="The corporation successfully sued a former employee for embezzlement, recovering all stolen assets."),
            Row(customer_id="C00999", article_snippet="The family trust was entirely exonerated following a two-year tax fraud investigation by the revenue service."),
            Row(customer_id="C01000", article_snippet="Local business owner opens third bakery location in the downtown district."),
            Row(customer_id="C00555", article_snippet="Local entrepreneur wins 'Business Leader of the Year' award for community investments and philanthropic efforts."),
        ]
        media_data = remap_customer_ids(media_data, customer_count)
        media_df = spark.createDataFrame(media_data)
        media_df.write.mode("overwrite").saveAsTable(f"{DATABASE}.adverse_media")
        print(f"Created {DATABASE}.adverse_media with {len(media_data)} records.")

        agent_text_data = [
            Row(customer_id="C00010", document_type="SWIFT_Note", text="Payment for consulting services. Beneficiary located in standard jurisdiction. Invoice attached."),
            Row(customer_id="C00015", document_type="SWIFT_Note", text="Urgent wire transfer. Purpose vague: 'Management fees'. Routing through three intermediary shell banks."),
            Row(customer_id="C00100", document_type="Corp_Registry", text="Ultimate Beneficial Owner (UBO) clearly identified as local retail operator. Transparent structure."),
            Row(customer_id="C00200", document_type="Corp_Registry", text="Complex ownership structure. UBO obfuscated behind two layers of offshore holding companies in high-risk jurisdictions."),
        ]
        agent_text_data = remap_customer_ids(agent_text_data, customer_count)
        notes_df = spark.createDataFrame(agent_text_data)
        notes_df.write.mode("overwrite").saveAsTable(f"{DATABASE}.unstructured_notes")
        print(f"Created {DATABASE}.unstructured_notes with {len(agent_text_data)} records.")

        print("Database setup complete. All tables are ready for querying.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()