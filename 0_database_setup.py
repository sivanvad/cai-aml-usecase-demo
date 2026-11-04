import os
import random
from pyspark.sql import SparkSession, Row

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


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, parsed)


def build_customer_rows(count: int = 5000):
    rows = []
    for i in range(1, count + 1):
        customer_id = f"C{i:05d}"
        entity_type = random.choices(["Individual", "Corporate"], weights=[0.85, 0.15])[0]
        age_years = random.randint(18, 85) if entity_type == "Individual" else random.randint(1, 30)
        country = random.choice(HIGH_RISK_COUNTRIES) if random.random() < 0.05 else random.choice(STANDARD_COUNTRIES)
        account_age_months = random.randint(1, 120)
        onboarding_channel = random.choices(
            ["In-Branch", "Digital-KYC", "Third-Party-Broker"],
            weights=[0.4, 0.5, 0.1],
        )[0]

        if entity_type == "Individual":
            income = round(random.uniform(30000, 250000), 2)
            cash_intensive = 0
        else:
            income = round(random.uniform(100000, 10000000), 2)
            cash_intensive = 1 if random.random() < 0.3 else 0

        pep_status = 1 if random.random() > 0.98 else 0
        occupation_risk = random.randint(1, 5)
        historic_risk = random.choice(["Low", "Medium", "High"])

        rows.append(
            Row(
                customer_id=customer_id,
                entity_type=entity_type,
                age_years=age_years,
                country_of_residence=country,
                account_age_months=account_age_months,
                onboarding_channel=onboarding_channel,
                declared_annual_income=income,
                cash_intensive_business=cash_intensive,
                pep_status=pep_status,
                occupation_risk=occupation_risk,
                historic_risk=historic_risk,
            )
        )
    return rows


def build_transaction_rows(count: int = 15000):
    rows = []
    for i in range(1, count + 1):
        customer_id = f"C{random.randint(1, 5000):05d}"
        tx_type = random.choices(TX_TYPES, weights=TX_WEIGHTS)[0]

        if tx_type == "International Wire":
            is_cross_border = 1
            amount = round(random.uniform(5000, 150000), 2)
        elif tx_type in {"Cash Deposit", "Cash Withdrawal"}:
            is_cross_border = 0
            amount = round(random.uniform(100, 9950), 2)
        elif tx_type == "Crypto Purchase":
            is_cross_border = random.choice([0, 1])
            amount = round(random.uniform(500, 25000), 2)
        else:
            is_cross_border = 1 if random.random() > 0.9 else 0
            amount = round(random.uniform(50, 75000), 2)

        rows.append(
            Row(
                tx_id=f"TX{i:07d}",
                customer_id=customer_id,
                amount=amount,
                is_cross_border=is_cross_border,
                transaction_type=tx_type,
            )
        )
    return rows


def build_network_rows(count: int = 5000):
    shared_ips = ["192.168.1.50", "10.0.0.99"]
    rows = []
    for i in range(1, count + 1):
        customer_id = f"C{i:05d}"
        ip = random.choice(shared_ips) if random.random() > 0.95 else f"192.168.1.{random.randint(100, 255)}"
        rows.append(Row(customer_id=customer_id, ip_address=ip))
    return rows


def write_table(spark, table_name, rows):
    df = spark.createDataFrame(rows)
    df.write.mode("overwrite").saveAsTable(f"aml_demo_db.{table_name}")
    print(f"Created aml_demo_db.{table_name} with {len(rows)} records.")


def main():
    customer_count = env_int("AML_CUSTOMER_COUNT", 5000)
    transaction_count = env_int("AML_TRANSACTION_COUNT", 15000)
    network_count = env_int("AML_NETWORK_COUNT", 5000)

    spark = SparkSession.builder.appName("AML_Database_Setup").enableHiveSupport().getOrCreate()
    try:
        spark.sql("CREATE DATABASE IF NOT EXISTS aml_demo_db")
        print("Database 'aml_demo_db' is ready.")

        write_table(spark, "customers", build_customer_rows(customer_count))
        write_table(spark, "transactions", build_transaction_rows(transaction_count))
        write_table(spark, "network_logs", build_network_rows(network_count))

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
        df_media = spark.createDataFrame(media_data)
        df_media.write.mode("overwrite").saveAsTable("aml_demo_db.adverse_media")
        print(f"Created aml_demo_db.adverse_media with {len(media_data)} records.")

        agent_text_data = [
            Row(
                customer_id="C00010",
                document_type="SWIFT_Note",
                text="Payment for consulting services. Beneficiary located in standard jurisdiction. Invoice attached.",
            ),
            Row(
                customer_id="C00015",
                document_type="SWIFT_Note",
                text="Urgent wire transfer. Purpose vague: 'Management fees'. Routing through three intermediary shell banks.",
            ),
            Row(
                customer_id="C00100",
                document_type="Corp_Registry",
                text="Ultimate Beneficial Owner (UBO) clearly identified as local retail operator. Transparent structure.",
            ),
            Row(
                customer_id="C00200",
                document_type="Corp_Registry",
                text="Complex ownership structure. UBO obfuscated behind two layers of offshore holding companies in high-risk jurisdictions.",
            ),
        ]
        df_agent_text = spark.createDataFrame(agent_text_data)
        df_agent_text.write.mode("overwrite").saveAsTable("aml_demo_db.unstructured_notes")
        print("Created aml_demo_db.unstructured_notes.")

        print("Database setup complete. All tables are ready for querying.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
