"""SQLite data store for market data."""

import sqlite3
from pathlib import Path


class SQLiteStore:
    """SQLite data store for market data."""

    def __init__(self, db_path='market_data.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        """Initialize database tables."""
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                timestamp REAL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        c.execute('''
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_time
            ON candles (symbol, timeframe)
        ''')
        self.conn.commit()

    def save_candles(self, symbol, timeframe, candles):
        """Save fetched candles to the database."""
        c = self.conn.cursor()
        for candle in candles:
            c.execute('''
                INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timeframe, candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]))
        self.conn.commit()

    def get_candles(self, symbol, timeframe='1m', since=None, limit=None):
        """Retrieve candles from the database."""
        c = self.conn.cursor()
        query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC
        '''
        params = [symbol.upper(), timeframe]
        if since:
            params.append(since)
        if limit:
            query += ' LIMIT ?'
            params.append(limit)
        c.execute(query, params)
        return c.fetchall()

    def close(self):
        """Close the database connection."""
        self.conn.close()
