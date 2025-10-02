import time
import pandas as pd
import logging
from statistics import mean
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "OnlineRetailDB"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEST_CUSTOMER_ID = None
TEST_INVOICE_ID = None
TEST_STOCK_CODE = None
REPEATS = 5

# ----------------------------------------------------------------------------------
# NOTE (Question 3 Clarification):
# The assignment asks for TWO MongoDB data modelling approaches. In this code:
#   1) Transaction‑centric (collection: invoices) -> each invoice embeds its items and a minimal customer reference
#   2) Customer‑centric (collection: customers)  -> each customer embeds invoices which embed their items
# SQLite portion removed per updated requirement – this file now benchmarks ONLY the two Mongo approaches.
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# DESIGN (High level)
# For performance comparison we measure each CRUD operation across only:
#   - Mongo (transaction‑centric)
#   - Mongo (customer‑centric)
# Improvements added:
#   * Multiple repeats with average timing to smooth noise
#   * Cleanup code excluded from timing sections where practical
#   * Ensured test documents contain fields needed for update/multiply operations (UnitPrice, Quantity)
#   * Added simple result normalization + formatted output
# ----------------------------------------------------------------------------------

def setup_test_data_mongo(db):
    """Locate an existing invoice + item + customer across both Mongo models.

    Tries transaction-centric first; if absent, attempts to derive from customer-centric.
    Raises if insufficient seed data.
    """
    global TEST_CUSTOMER_ID, TEST_INVOICE_ID, TEST_STOCK_CODE
    invoice_doc = db.invoices.find_one({"items.0": {"$exists": True}})
    if invoice_doc:
        TEST_INVOICE_ID = invoice_doc.get("_id")
        TEST_CUSTOMER_ID = invoice_doc.get("customer", {}).get("id")
        if invoice_doc.get("items"):
            TEST_STOCK_CODE = invoice_doc["items"][0].get("StockCode")
    else:
        cust_doc = db.customers.find_one({"invoices.0.items.0": {"$exists": True}})
        if cust_doc:
            TEST_CUSTOMER_ID = cust_doc.get("_id")
            first_invoice = cust_doc.get("invoices", [])[0]
            TEST_INVOICE_ID = first_invoice.get("invoiceNo") if first_invoice else None
            if first_invoice and first_invoice.get("items"):
                TEST_STOCK_CODE = first_invoice["items"][0].get("StockCode")

    if not all([TEST_CUSTOMER_ID, TEST_INVOICE_ID, TEST_STOCK_CODE]):
        raise RuntimeError("Could not derive test identifiers from MongoDB. Ensure data loaded via mongo_insert.py")
    logging.info(f"Using Test Data: CustomerID={TEST_CUSTOMER_ID}, InvoiceID={TEST_INVOICE_ID}, StockCode={TEST_STOCK_CODE}")

# --- SQL CRUD Functions ---

#############################
# Removed SQL helper methods (no-op placeholders retained if referenced earlier)
#############################

#############################
# SQL bulk create removed
#############################

def read_single_sql(conn):
    pd.read_sql("SELECT * FROM invoice_items WHERE InvoiceNo = ?", conn, params=(TEST_INVOICE_ID,))

def read_bulk_sql(conn):
    pd.read_sql("""
        SELECT * FROM invoices i JOIN invoice_items ii ON i.InvoiceNo = ii.InvoiceNo
        WHERE i.CustomerID = ?
    """, conn, params=(TEST_CUSTOMER_ID,))

def update_single_sql(conn):
    cursor = conn.cursor()
    cursor.execute("UPDATE invoice_items SET Quantity = 99 WHERE InvoiceNo = ? AND StockCode = ?",
                   (TEST_INVOICE_ID, TEST_STOCK_CODE))
    conn.commit()

def update_bulk_sql(conn):
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE invoice_items SET UnitPrice = UnitPrice * 0.9
        WHERE InvoiceNo IN (SELECT InvoiceNo FROM invoices WHERE CustomerID = ?)
    """, (TEST_CUSTOMER_ID,))
    conn.commit()

#############################
# SQL delete single removed
#############################

def delete_bulk_sql(conn):
    cursor = conn.cursor()
    # In a real scenario, you'd backup before deleting. Here we just perform the delete.
    cursor.execute("DELETE FROM invoice_items WHERE InvoiceNo = ?", (TEST_INVOICE_ID,))
    cursor.execute("DELETE FROM invoices WHERE InvoiceNo = ?", (TEST_INVOICE_ID,))
    conn.commit()

############################################################
# MongoDB Transaction-Centric CRUD (collection: invoices)
############################################################

def create_single_mongo_t(db):
    db.invoices.update_one(
        {"_id": TEST_INVOICE_ID},
        {"$push": {"items": {"StockCode": "TEST01", "Quantity": 1, "UnitPrice": 9.99}}}
    )

def cleanup_create_single_mongo_t(db):
    db.invoices.update_one({"_id": TEST_INVOICE_ID}, {"$pull": {"items": {"StockCode": "TEST01"}}})

def create_bulk_mongo_t(db):
    new_invoice = {
        "_id": "INVOICE_TEMP",
        "customer": {"id": TEST_CUSTOMER_ID},
        "items": [
            {"StockCode": f"BULK{i}", "Quantity": i, "UnitPrice": round(i * 1.5 + 0.05, 2)}
            for i in range(10)
        ]
    }
    db.invoices.insert_one(new_invoice)

def cleanup_create_bulk_mongo_t(db):
    db.invoices.delete_one({"_id": "INVOICE_TEMP"})

def read_single_mongo_t(db):
    db.invoices.find_one({"_id": TEST_INVOICE_ID})

def read_bulk_mongo_t(db):
    list(db.invoices.find({"customer.id": TEST_CUSTOMER_ID}))

def update_single_mongo_t(db):
    db.invoices.update_one({"_id": TEST_INVOICE_ID, "items.StockCode": TEST_STOCK_CODE}, {"$set": {"items.$.Quantity": 99}})

def update_bulk_mongo_t(db):
    db.invoices.update_many({"customer.id": TEST_CUSTOMER_ID}, {"$mul": {"items.$[].UnitPrice": 0.9}})

def delete_single_mongo_t(db):
    item_to_remove = {"StockCode": TEST_STOCK_CODE}
    db.invoices.update_one({"_id": TEST_INVOICE_ID}, {"$pull": {"items": item_to_remove}})

def cleanup_delete_single_mongo_t(db):
    # Recreate with minimal fields (Quantity/UnitPrice may exist or not originally)
    db.invoices.update_one(
        {"_id": TEST_INVOICE_ID},
        {"$addToSet": {"items": {"StockCode": TEST_STOCK_CODE, "Quantity": 1, "UnitPrice": 1.0}}}
    )

def delete_bulk_mongo_t(db):
    # Delete whole invoice document (restored by cleanup)
    global _backup_invoice_doc
    _backup_invoice_doc = db.invoices.find_one({"_id": TEST_INVOICE_ID})
    db.invoices.delete_one({"_id": TEST_INVOICE_ID})

def cleanup_delete_bulk_mongo_t(db):
    global _backup_invoice_doc
    if _backup_invoice_doc:
        db.invoices.insert_one(_backup_invoice_doc)
    _backup_invoice_doc = None

############################################################
# MongoDB Customer-Centric CRUD (collection: customers)
############################################################

def create_single_mongo_c(db):
    db.customers.update_one(
        {"_id": TEST_CUSTOMER_ID, "invoices.invoiceNo": TEST_INVOICE_ID},
        {"$push": {"invoices.$.items": {"StockCode": "TEST01", "Quantity": 1, "UnitPrice": 9.99}}}
    )

def cleanup_create_single_mongo_c(db):
    db.customers.update_one(
        {"_id": TEST_CUSTOMER_ID, "invoices.invoiceNo": TEST_INVOICE_ID},
        {"$pull": {"invoices.$.items": {"StockCode": "TEST01"}}}
    )

def create_bulk_mongo_c(db):
    new_invoice = {
        "invoiceNo": "INVOICE_C_TEMP",
        "items": [
            {"StockCode": f"BULK{i}", "Quantity": i, "UnitPrice": round(i * 1.5 + 0.05, 2)}
            for i in range(10)
        ]
    }
    db.customers.update_one({"_id": TEST_CUSTOMER_ID}, {"$push": {"invoices": new_invoice}})

def cleanup_create_bulk_mongo_c(db):
    db.customers.update_one(
        {"_id": TEST_CUSTOMER_ID},
        {"$pull": {"invoices": {"invoiceNo": "INVOICE_C_TEMP"}}}
    )

def read_single_mongo_c(db):
    db.customers.find_one({"_id": TEST_CUSTOMER_ID, "invoices.invoiceNo": TEST_INVOICE_ID}, {"invoices.$": 1})

def read_bulk_mongo_c(db):
    db.customers.find_one({"_id": TEST_CUSTOMER_ID})

def update_single_mongo_c(db):
    db.customers.update_one({"_id": TEST_CUSTOMER_ID}, {"$set": {"invoices.$[inv].items.$[item].Quantity": 99}},
                          array_filters=[{"inv.invoiceNo": TEST_INVOICE_ID}, {"item.StockCode": TEST_STOCK_CODE}])

def update_bulk_mongo_c(db):
    db.customers.update_one({"_id": TEST_CUSTOMER_ID}, {"$mul": {"invoices.$[].items.$[].UnitPrice": 0.9}})

def delete_single_mongo_c(db):
    item_to_remove = {"StockCode": TEST_STOCK_CODE}
    db.customers.update_one(
        {"_id": TEST_CUSTOMER_ID, "invoices.invoiceNo": TEST_INVOICE_ID},
        {"$pull": {"invoices.$.items": item_to_remove}}
    )

def cleanup_delete_single_mongo_c(db):
    db.customers.update_one(
        {"_id": TEST_CUSTOMER_ID, "invoices.invoiceNo": TEST_INVOICE_ID},
        {"$addToSet": {"invoices.$.items": {"StockCode": TEST_STOCK_CODE, "Quantity": 1, "UnitPrice": 1.0}}}
    )

def delete_bulk_mongo_c(db):
    global _backup_invoice_embedded
    customer_doc = db.customers.find_one(
        {"_id": TEST_CUSTOMER_ID},
        {"invoices": {"$elemMatch": {"invoiceNo": TEST_INVOICE_ID}}}
    )
    _backup_invoice_embedded = None
    if customer_doc and customer_doc.get('invoices'):
        _backup_invoice_embedded = customer_doc['invoices'][0]
    db.customers.update_one({"_id": TEST_CUSTOMER_ID}, {"$pull": {"invoices": {"invoiceNo": TEST_INVOICE_ID}}})

def cleanup_delete_bulk_mongo_c(db):
    global _backup_invoice_embedded
    if _backup_invoice_embedded:
        db.customers.update_one(
            {"_id": TEST_CUSTOMER_ID},
            {"$addToSet": {"invoices": _backup_invoice_embedded}}
        )
    _backup_invoice_embedded = None

############################################################
# Benchmark Helpers
############################################################

def _time_callable(fn, *args, repeats=REPEATS, cleanup=None):
    """Time a callable over several repeats, optionally calling a cleanup function after each run.

    The cleanup is NOT timed. Returns average milliseconds.
    """
    durations = []
    for _ in range(repeats):
        start = time.perf_counter(); fn(*args); durations.append((time.perf_counter() - start) * 1000)
        if cleanup:
            try:
                cleanup(*args)
            except Exception as ce:  # non-fatal cleanup error
                logging.warning(f"Cleanup failed for {fn.__name__}: {ce}")
    return mean(durations)

# --- Main Execution ---

def main():
    results = {}
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.admin.command('ping')
        mongo_db = mongo_client[MONGO_DB_NAME]
        logging.info("Mongo connection successful.")
    except ConnectionFailure as e:
        logging.error(f"Mongo connection failed: {e}")
        return

    try:
        setup_test_data_mongo(mongo_db)
        scenarios = {
            "Create Single": (create_single_mongo_t, cleanup_create_single_mongo_t,
                               create_single_mongo_c, cleanup_create_single_mongo_c),
            "Create Bulk":   (create_bulk_mongo_t, cleanup_create_bulk_mongo_t,
                               create_bulk_mongo_c, cleanup_create_bulk_mongo_c),
            "Read Single":   (read_single_mongo_t, None,
                               read_single_mongo_c, None),
            "Read Bulk":     (read_bulk_mongo_t, None,
                               read_bulk_mongo_c, None),
            "Update Single": (update_single_mongo_t, None,
                               update_single_mongo_c, None),
            "Update Bulk":   (update_bulk_mongo_t, None,
                               update_bulk_mongo_c, None),
            "Delete Single": (delete_single_mongo_t, cleanup_delete_single_mongo_t,
                               delete_single_mongo_c, cleanup_delete_single_mongo_c),
            "Delete Bulk":   (delete_bulk_mongo_t, cleanup_delete_bulk_mongo_t,
                               delete_bulk_mongo_c, cleanup_delete_bulk_mongo_c),
        }

        for name, (t_fn, t_cleanup, c_fn, c_cleanup) in scenarios.items():
            trans_time = _time_callable(t_fn, mongo_db, cleanup=t_cleanup)
            cust_time = _time_callable(c_fn, mongo_db, cleanup=c_cleanup)
            results[name] = (trans_time, cust_time)
    finally:
        mongo_client.close()
        logging.info("Mongo connection closed.")

    print(f"\n--- Mongo CRUD Performance (avg of {REPEATS} runs, ms) ---")
    print(f"{'Operation':<15} | {'Transaction-Centric':>20} | {'Customer-Centric':>20} | {'Faster':>8}")
    print('-' * 75)
    for name, (t_time, c_time) in results.items():
        faster = 'Trans' if t_time < c_time else ('Cust' if c_time < t_time else 'Tie')
        print(f"{name:<15} | {t_time:>20.2f} | {c_time:>20.2f} | {faster:>8}")

if __name__ == "__main__":
    main()