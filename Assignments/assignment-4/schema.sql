PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS customers (
  CustomerID TEXT PRIMARY KEY,
  Country TEXT
);

CREATE TABLE IF NOT EXISTS products (
  StockCode TEXT PRIMARY KEY,
  Description TEXT
);

CREATE TABLE IF NOT EXISTS invoices (
  InvoiceNo TEXT PRIMARY KEY,
  InvoiceDate TIMESTAMP,
  CustomerID TEXT,
  FOREIGN KEY (CustomerID) REFERENCES customers(CustomerID)
);

CREATE TABLE IF NOT EXISTS invoice_items (
  InvoiceNo TEXT,
  StockCode TEXT,
  Quantity INTEGER,
  UnitPrice REAL,
  PRIMARY KEY (InvoiceNo, StockCode),
  FOREIGN KEY (InvoiceNo) REFERENCES invoices(InvoiceNo),
  FOREIGN KEY (StockCode) REFERENCES products(StockCode)
);