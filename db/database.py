import sqlite3
from pathlib import Path
import json

DB_PATH = "invoice_db.sqlite"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Table for storing final extracted invoices
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT,
            vendor_address TEXT,
            vendor_phone TEXT,
            invoice_number TEXT,
            invoice_date TEXT,
            due_date TEXT,
            buyer_name TEXT,
            buyer_address TEXT,
            buyer_phone TEXT,
            payment_details TEXT,
            subtotal REAL,
            tax REAL,
            tax_rate TEXT,
            total REAL,
            currency TEXT,
            payment_terms TEXT,
            notes TEXT,
            overall_confidence REAL,
            UNIQUE(vendor_name, invoice_number)
        )
    """)

    # Table for line items linked to invoices
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS line_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_id INTEGER,
            description TEXT,
            quantity REAL,
            unit_price REAL,
            total REAL,
            confidence REAL,
            FOREIGN KEY (invoice_id) REFERENCES invoices (id) ON DELETE CASCADE
        )
    """)

    # Table for Stage 1 Uniqueness tracking (File Hashing)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT UNIQUE,
            file_name TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            invoice_id INTEGER,
            FOREIGN KEY (invoice_id) REFERENCES invoices (id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()

def check_file_hash_exists(file_hash: str) -> bool:
    """Check if a file hash has already been processed."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM processed_files WHERE file_hash = ?", (file_hash,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def record_processed_file(file_hash: str, file_name: str, invoice_id: int = None):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO processed_files (file_hash, file_name, invoice_id) VALUES (?, ?, ?)",
            (file_hash, file_name, invoice_id)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Already exists
    finally:
        conn.close()

def insert_invoice(extracted_data, file_hash: str = None, file_name: str = None) -> tuple[bool, str]:
    """
    Inserts a new invoice. Enforces (vendor_name, invoice_number) uniqueness.
    Returns (success_boolean, message_or_error)
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO invoices (
                vendor_name, vendor_address, vendor_phone, invoice_number,
                invoice_date, due_date, buyer_name, buyer_address, buyer_phone,
                payment_details, subtotal, tax, tax_rate, total, currency,
                payment_terms, notes, overall_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            extracted_data.vendor_name, extracted_data.vendor_address, extracted_data.vendor_phone,
            extracted_data.invoice_number, extracted_data.invoice_date, extracted_data.due_date,
            extracted_data.buyer_name, extracted_data.buyer_address, extracted_data.buyer_phone,
            extracted_data.payment_details, extracted_data.subtotal, extracted_data.tax,
            extracted_data.tax_rate, extracted_data.total, extracted_data.currency,
            extracted_data.payment_terms, extracted_data.notes, extracted_data.overall_confidence
        ))
        
        invoice_id = cursor.lastrowid

        for item in extracted_data.line_items:
            cursor.execute("""
                INSERT INTO line_items (invoice_id, description, quantity, unit_price, total, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                invoice_id, item.description, item.quantity, item.unit_price, item.total, item.confidence
            ))
            
        if file_hash:
            record_processed_file(file_hash, file_name or "unknown", invoice_id)

        conn.commit()
        return True, f"Invoice {extracted_data.invoice_number} successfully stored with ID {invoice_id}."
        
    except sqlite3.IntegrityError:
        conn.rollback()
        return False, f"Logical Duplicate: An invoice for '{extracted_data.vendor_name}' with number '{extracted_data.invoice_number}' already exists."
    except Exception as e:
        conn.rollback()
        return False, str(e)
    finally:
        conn.close()
