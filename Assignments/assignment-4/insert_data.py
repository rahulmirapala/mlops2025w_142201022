import sqlite3
import pandas as pd
import os

DATABASE_NAME = 'online_retail.db'
DATASET_PATH = os.path.join('data', 'Online Retail.xlsx')
TARGET_ITEM_RECORDS = 1000
SCHEMA_PATH = 'schema.sql' 

def create_database_schema(cursor):
    print(f"Applying schema from '{SCHEMA_PATH}'...")
    try:
        with open(SCHEMA_PATH, 'r') as f:
            sql_script = f.read()
        cursor.executescript(sql_script)
        print("Schema applied successfully.")
    except FileNotFoundError:
        print(f"Error: The schema file '{SCHEMA_PATH}' was not found.")
        raise

def clean_data(df):
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


def main():
    try:
        df_all = pd.read_excel(DATASET_PATH)
        df_cleaned = clean_data(df_all)

        unique_invoices = df_cleaned['InvoiceNo'].unique()

        sampled_invoice_nos = []
        df_sampled = pd.DataFrame()

        while len(df_sampled) < TARGET_ITEM_RECORDS and len(sampled_invoice_nos) < len(unique_invoices):
            more_invoices = (
                pd.Series(unique_invoices)
                .sample(n=200, random_state=None)
                .tolist()
            )
            sampled_invoice_nos = list(set(sampled_invoice_nos + more_invoices))
            df_sampled = df_cleaned[df_cleaned['InvoiceNo'].isin(sampled_invoice_nos)]

        customers = df_sampled[['CustomerID', 'Country']].drop_duplicates(subset=['CustomerID'])
        products = df_sampled[['StockCode', 'Description']].drop_duplicates(subset=['StockCode'])
        invoices = df_sampled[['InvoiceNo', 'InvoiceDate', 'CustomerID']].drop_duplicates(subset=['InvoiceNo'])
        invoice_items = df_sampled[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice']].drop_duplicates(
            subset=['InvoiceNo', 'StockCode']
        )

        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            create_database_schema(cursor)
            cursor.execute("DELETE FROM invoice_items;")
            cursor.execute("DELETE FROM invoices;")
            cursor.execute("DELETE FROM products;")
            cursor.execute("DELETE FROM customers;")
            conn.commit()

            customers.to_sql('customers', conn, if_exists='append', index=False)
            products.to_sql('products', conn, if_exists='append', index=False)
            invoices.to_sql('invoices', conn, if_exists='append', index=False)
            invoice_items.to_sql('invoice_items', conn, if_exists='append', index=False)

            conn.commit()

            customer_count = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            product_count = cursor.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            invoice_count = cursor.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
            item_count = cursor.execute("SELECT COUNT(*) FROM invoice_items").fetchone()[0]

    except FileNotFoundError:
        print(f"Error: The file '{DATASET_PATH}' was not found.")
        print("Please ensure the dataset is in the 'data/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
