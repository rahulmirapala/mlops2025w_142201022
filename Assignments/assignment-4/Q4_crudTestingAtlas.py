import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import random
from datetime import datetime

MONGO_URI = "mongodb+srv://rahulmirapala:Rahul%401410@rahul.intqf7h.mongodb.net/"
DB_NAME = "OnlineRetailDB"
COLLECTION_NAME = "invoices"

# Helper functions
def get_random_invoice_id(collection):
    try:
        random_doc = list(collection.aggregate([{"$sample": {"size": 1}}]))
        if random_doc:
            return random_doc[0]['_id']
    except Exception as e:
        print(f"Could not fetch a random document: {e}")
    return None

def time_operation(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Operation took: {duration:.6f} seconds.")
    return result

def test_create_operation(collection):
    print("\nTesting CREATE")
    test_doc = {
        "_id": f"TEST_INVOICE_{int(time.time())}", 
        "invoiceDate": datetime.utcnow(),
        "customer": {"id": "99999", "country": "Testland"},
        "items": [
            {"StockCode": "T01", "Description": "Test Product A", "Quantity": 5, "UnitPrice": 9.99},
            {"StockCode": "T02", "Description": "Test Product B", "Quantity": 1, "UnitPrice": 25.50}
        ]
    }
    print(f"Inserting new document with ID: {test_doc['_id']}")
    time_operation(collection.insert_one, test_doc)
    return test_doc['_id']

def test_read_operation(collection, invoice_id):
    print(f"\nTesting READ")
    print(f"Searching for document with ID: {invoice_id}")
    document = time_operation(collection.find_one, {"_id": invoice_id})
    if document:
        print("Document found successfully")
    else:
        print("Document not found")

def test_update_operation(collection, invoice_id):
    print(f"\nTesting UPDATE")
    print(f"Updating the country for invoice ID: {invoice_id}")
    update_query = {"$set": {"customer.country": "UnitTest Country"}}
    result = time_operation(collection.update_one, {"_id": invoice_id}, update_query)
    if result.modified_count > 0:
        print("Document updated successfully.")
    else:
        print("Document could not be updated.")

def test_delete_operation(collection, invoice_id):
    print(f"\nTesting DELETE")
    print(f"Deleting document with ID: {invoice_id}")
    result = time_operation(collection.delete_one, {"_id": invoice_id})
    if result.deleted_count > 0:
        print("Document deleted successfully (cleanup complete).")
    else:
        print("Document could not be deleted.")

def main():
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas.")
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
    except (ConnectionFailure, OperationFailure) as e:
        print(f"Error: Could not connect to or authenticate with MongoDB.\n{e}")
        return
    
    print("\nReading a random document from existing data")
    random_id = get_random_invoice_id(collection)
    if random_id:
        test_read_operation(collection, random_id) # read test
    else:
        print("Data not found in the collection")

    temp_doc_id = test_create_operation(collection) # create test
    if temp_doc_id:
        test_update_operation(collection, temp_doc_id) # update test
        test_delete_operation(collection, temp_doc_id) # delete test

    print("\n All tests complete.")
    client.close()

if __name__ == "__main__":
    main()
