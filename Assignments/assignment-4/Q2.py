import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os

DATASET_PATH = os.path.join('data', 'Online Retail.xlsx')
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "OnlineRetailDB"
SAMPLE_SIZE = 5000  

def clean_data(df):
    df.dropna(subset=['CustomerID', 'InvoiceNo'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


def insert_transaction_centric(db, data):
    collection = db["invoices"]
    collection.drop()  
    invoices_to_insert = []

    for invoice_no, group in data.groupby('InvoiceNo'):
        customer_info = group[['CustomerID', 'Country']].iloc[0]
        invoice_doc = {
            "_id": str(invoice_no),
            "invoiceDate": group['InvoiceDate'].iloc[0].to_pydatetime(),
            "customer": {
                "id": customer_info['CustomerID'],
                "country": customer_info['Country']
            },
            "items": group[['StockCode', 'Description', 'Quantity', 'UnitPrice']].to_dict('records')
        }
        invoices_to_insert.append(invoice_doc)

    if invoices_to_insert:
        collection.insert_many(invoices_to_insert)


def insert_customer_centric(db, data):
    collection = db["customers"]
    collection.drop() 
    customers_to_insert = []

    for customer_id, group in data.groupby('CustomerID'):
        customer_doc = {
            "_id": str(customer_id),
            "country": group['Country'].iloc[0],
            "invoices": []
        }

        for invoice_no, invoice_group in group.groupby('InvoiceNo'):
            invoice_data = {
                "invoiceNo": str(invoice_no),
                "invoiceDate": invoice_group['InvoiceDate'].iloc[0].to_pydatetime(),
                "items": invoice_group[['StockCode', 'Description', 'Quantity', 'UnitPrice']].to_dict('records')
            }
            customer_doc["invoices"].append(invoice_data)

        customers_to_insert.append(customer_doc)

    if customers_to_insert:
        collection.insert_many(customers_to_insert)


def main():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
    except ConnectionFailure as e:
        return

    db = client[DB_NAME]

    df_all = pd.read_excel(DATASET_PATH)
    df_cleaned = clean_data(df_all)

    df_sampled = df_cleaned.sample(n=SAMPLE_SIZE, random_state=42)

    insert_transaction_centric(db, df_sampled)
    insert_customer_centric(db, df_sampled)

    client.close()


if __name__ == "__main__":
    main()
