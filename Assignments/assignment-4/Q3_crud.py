import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import time
import os

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "OnlineRetailDB"
TRANSACTION_COLLECTION_NAME = "invoices"
CUSTOMER_COLLECTION_NAME = "customers"
DATASET_PATH = os.path.join('data', 'Online Retail.xlsx')
BULK_OPERATION_COUNT = 100 
TEST_COUNTRY = "United Kingdom"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("MongoDB connection successful.")
    db = client[DB_NAME]
    transactions_collection = db[TRANSACTION_COLLECTION_NAME]
    customers_collection = db[CUSTOMER_COLLECTION_NAME]
except ConnectionFailure as e:
    print(f"Could not connect to MongoDB: {e}")
    exit()

try:
    random_invoice_id = transactions_collection.find_one({}, {"_id": 1})['_id']
    random_customer_id = customers_collection.find_one({}, {"_id": 1})['_id']
except (TypeError, IndexError):
    print("Error: Collections are empty")
    exit()

df_all = pd.read_excel(DATASET_PATH)
df_all.dropna(subset=['CustomerID', 'InvoiceNo', 'Description', 'Country'], inplace=True)
df_all['CustomerID'] = df_all['CustomerID'].astype(int).astype(str)
new_record_df = df_all.sample(n=1)
new_invoice_id = "TEST_INV_001"
new_customer_id = "TEST_CUST_001"
    
def time_operation(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return duration, result

# CREATE
def create_single_transaction():
    doc = {"_id": new_invoice_id, "customer": {"id": new_record_df['CustomerID'].iloc[0]}}
    transactions_collection.insert_one(doc)
    transactions_collection.delete_one({"_id": new_invoice_id})

def create_bulk_transactions():
    docs = [{"_id": f"BULK_INV_{i}"} for i in range(BULK_OPERATION_COUNT)]
    transactions_collection.insert_many(docs)
    transactions_collection.delete_many({"_id": {"$regex": "^BULK_INV_"}})

def create_single_customer():
    doc = {"_id": new_customer_id, "country": new_record_df['Country'].iloc[0]}
    customers_collection.insert_one(doc)
    customers_collection.delete_one({"_id": new_customer_id})

def create_bulk_customers():
    docs = [{"_id": f"BULK_CUST_{i}"} for i in range(BULK_OPERATION_COUNT)]
    customers_collection.insert_many(docs)
    customers_collection.delete_many({"_id": {"$regex": "^BULK_CUST_"}})
    
# READ
def read_single_transaction():
    transactions_collection.find_one({"_id": random_invoice_id})

def read_single_customer():
    customers_collection.find_one({"_id": random_customer_id})

# UPDATE
def update_single_transaction():
    transactions_collection.update_one({"_id": random_invoice_id}, {"$set": {"customer.note": "single_update"}})
    transactions_collection.update_one({"_id": random_invoice_id}, {"$unset": {"customer.note": ""}}) # Cleanup

def update_bulk_transactions_by_country():
    transactions_collection.update_many(
        {"customer.country": TEST_COUNTRY},
        {"$set": {"customer.note": "bulk_update"}}
    )
    # Cleanup the change
    transactions_collection.update_many(
        {"customer.country": TEST_COUNTRY},
        {"$unset": {"customer.note": ""}}
    )

def update_single_customer():
    customers_collection.update_one({"_id": random_customer_id}, {"$set": {"note": "single_update"}})
    customers_collection.update_one({"_id": random_customer_id}, {"$unset": {"note": ""}}) # Cleanup

def update_bulk_customers_by_country():
    customers_collection.update_many({"country": TEST_COUNTRY}, {"$set": {"note": "bulk_update"}})
    customers_collection.update_many({"country": TEST_COUNTRY}, {"$unset": {"note": ""}}) # Cleanup

# DELETE
def delete_single_transaction():
    temp_id = "TEMP_DEL_INV_001"
    transactions_collection.insert_one({"_id": temp_id})
    transactions_collection.delete_one({"_id": temp_id})

def delete_bulk_transactions_by_country():
    docs = [{"_id": f"TEMP_BULK_DEL_{i}", "customer": {"country": "Testland"}} for i in range(BULK_OPERATION_COUNT)]
    transactions_collection.insert_many(docs)
    transactions_collection.delete_many({"customer.country": "Testland"})

def delete_single_customer():
    temp_id = "TEMP_DEL_CUST_001"
    customers_collection.insert_one({"_id": temp_id})
    customers_collection.delete_one({"_id": temp_id})

def delete_bulk_customers_by_country():
    docs = [{"_id": f"TEMP_BULK_DEL_CUST_{i}", "country": "Testland"} for i in range(BULK_OPERATION_COUNT)]
    customers_collection.insert_many(docs)
    customers_collection.delete_many({"country": "Testland"})

if __name__ == "__main__":
    results = {}

    # CREATE
    results["Create Single Transaction"], _ = time_operation(create_single_transaction)
    results["Create Single Customer"], _ = time_operation(create_single_customer)
    results[f"Create Bulk ({BULK_OPERATION_COUNT}) Transaction"], _ = time_operation(create_bulk_transactions)
    results[f"Create Bulk ({BULK_OPERATION_COUNT}) Customer"], _ = time_operation(create_bulk_customers)

    # READ
    results["Read Single Transaction (by ID)"], _ = time_operation(read_single_transaction)
    results["Read Single Customer (by ID)"], _ = time_operation(read_single_customer)

    # UPDATE
    results["Update Single Transaction"], _ = time_operation(update_single_transaction)
    results["Update Single Customer"], _ = time_operation(update_single_customer)
    results[f"Update Bulk Transactions (by Country)"], _ = time_operation(update_bulk_transactions_by_country)
    results[f"Update Bulk Customers (by Country)"], _ = time_operation(update_bulk_customers_by_country)
    
    # DELETE
    results["Delete Single Transaction"], _ = time_operation(delete_single_transaction)
    results["Delete Single Customer"], _ = time_operation(delete_single_customer)
    results[f"Delete Bulk Transactions (by Country)"], _ = time_operation(delete_bulk_transactions_by_country)
    results[f"Delete Bulk Customers (by Country)"], _ = time_operation(delete_bulk_customers_by_country)

    print("\nPERFORMANCE RESULTS (in milliseconds)")
    for op, duration in results.items():
        duration_ms = duration * 1000
        print(f"{op:<45}: {duration_ms:.3f} ms")

    client.close()
