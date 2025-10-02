import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import os

MONGO_URI = "mongodb+srv://rahulmirapala:Rahul%401410@rahul.intqf7h.mongodb.net/"
DB_NAME = "OnlineRetailDB"
COLLECTION_NAME = "invoices" # This will store the transaction-centric data
DATASET_PATH = os.path.join('data', 'Online Retail.xlsx')
SAMPLE_SIZE = 1000

def clean_data(df):
    df.dropna(subset=['CustomerID', 'InvoiceNo', 'Description'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def convert_to_transaction_centric(data):
    invoices_to_insert = []
    for invoice_no, group in data.groupby('InvoiceNo'):
        customer_info = group[['CustomerID', 'Country']].iloc[0]
        invoice_doc = {
            "_id": invoice_no,
            "invoiceDate": group['InvoiceDate'].iloc[0].to_pydatetime(),
            "customer": {
                "id": customer_info['CustomerID'],
                "country": customer_info['Country']
            },
            "items": group[['StockCode', 'Description', 'Quantity', 'UnitPrice']].to_dict('records')
        }
        invoices_to_insert.append(invoice_doc)
    return invoices_to_insert

def main():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas.")
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
    except ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB. Check your connection string and network settings.\n{e}")
        return
    except OperationFailure as e:
        print(f"Error: Authentication failed. Please check your username and password.\n{e}")
        return

    try:
        df_all = pd.read_excel(DATASET_PATH)
        df_cleaned = clean_data(df_all)
        unique_invoices = df_cleaned['InvoiceNo'].unique()
        df_sampled = df_cleaned[df_cleaned['InvoiceNo'].isin(unique_invoices[:500])].head(SAMPLE_SIZE)
    except FileNotFoundError:
        print(f"Error: The file '{DATASET_PATH}' was not found. Please make sure it's in a 'data' subfolder.")
        return
    except Exception as e:
        print(f"An error occurred during data loading and cleaning: {e}")
        return

    transactional_docs = convert_to_transaction_centric(df_sampled)

    if transactional_docs:
        try:
            collection.delete_many({})
            collection.insert_many(transactional_docs)
            print("\nData insertion successful!")
        except Exception as e:
            print(f"An error occurred during data insertion: {e}")
    else:
        print("No documents to insert.")

    client.close()
    print("Connection to MongoDB closed.")

if __name__ == "__main__":
    main()


