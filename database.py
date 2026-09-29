import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "fraud_log.db"

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            transaction_id TEXT,
            amount REAL,
            account_from TEXT,
            account_to TEXT,
            anomaly_score REAL,
            risk_level TEXT,
            status TEXT DEFAULT 'Pending Review'
        )
    ''')
    conn.commit()
    conn.close()

def log_anomaly(transaction_data, score, risk_level):
    """Log a single anomalous transaction into the database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Extract details (adjust keys based on your actual column names)
    c.execute('''
        INSERT INTO anomalies (timestamp, transaction_id, amount, account_from, account_to, anomaly_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(transaction_data.get('transaction_id', 'Unknown')),
        float(transaction_data.get('amount', 0)),
        str(transaction_data.get('nameOrig', 'Unknown')),
        str(transaction_data.get('nameDest', 'Unknown')),
        float(score),
        risk_level
    ))
    
    conn.commit()
    conn.close()

def get_all_logs():
    """Retrieve all logs for the dashboard."""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM anomalies ORDER BY id DESC", conn)
    conn.close()
    return df

def clear_logs():
    """Optional: Clear logs for a fresh demo."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM anomalies")
    conn.commit()
    conn.close()