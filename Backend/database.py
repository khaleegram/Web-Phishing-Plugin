import sqlite3

DB_FILE = "dataset/phishing_data.db"

def initialize_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS phishing_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            phishing INTEGER,
            confidence FLOAT
        )
    ''')
    conn.commit()
    conn.close()

def save_data(url, phishing, confidence):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO phishing_data (url, phishing, confidence) VALUES (?, ?, ?)
    ''', (url, phishing, confidence))
    conn.commit()
    conn.close()

# Initialize the database
initialize_database()
