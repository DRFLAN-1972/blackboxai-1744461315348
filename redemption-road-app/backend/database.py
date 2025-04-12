import sqlite3
from contextlib import contextmanager
from datetime import datetime
import logging
from typing import Any, Generator

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass

class DatabaseConnection:
    def __init__(self, db_path: str = "database/music.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def __enter__(self) -> 'DatabaseConnection':
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self._ensure_tables_exist()
            return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def _ensure_tables_exist(self) -> None:
        """Create necessary tables if they don't exist."""
        self.execute("""
            CREATE TABLE IF NOT EXISTS isrc_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'retired', 'reserved'))
            )
        """)
        
        self.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                artist_id INTEGER NOT NULL,
                isrc_code TEXT UNIQUE NOT NULL,
                genre TEXT NOT NULL,
                duration INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                release_date TIMESTAMP,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'released', 'archived')),
                FOREIGN KEY (isrc_code) REFERENCES isrc_codes(code),
                FOREIGN KEY (artist_id) REFERENCES users(id)
            )
        """)
        
        self.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                package TEXT NOT NULL CHECK(package IN ('bronze', 'silver', 'gold', 'platinum', 'doublePlatinum')),
                songs_generated INTEGER DEFAULT 0,
                clones_created INTEGER DEFAULT 0,
                subscription_start TIMESTAMP NOT NULL,
                subscription_end TIMESTAMP NOT NULL,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'suspended', 'cancelled'))
            )
        """)
        
        self.execute("""
            CREATE TABLE IF NOT EXISTS distribution_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL,
                platform TEXT NOT NULL,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'distributed', 'failed')),
                distributed_at TIMESTAMP,
                platform_url TEXT,
                FOREIGN KEY (song_id) REFERENCES songs(id)
            )
        """)
        
        self.execute("""
            CREATE TABLE IF NOT EXISTS radio_submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL,
                submission_date TIMESTAMP NOT NULL,
                format TEXT NOT NULL,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'submitted', 'accepted', 'rejected')),
                feedback TEXT,
                FOREIGN KEY (song_id) REFERENCES songs(id)
            )
        """)

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL query with error handling."""
        try:
            return self.cursor.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"Database error executing query: {e}")
            raise DatabaseError(f"Database operation failed: {e}")

    def fetch_one(self, query: str, params: tuple = ()) -> tuple:
        """Execute query and fetch one result."""
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            logger.error(f"Database error fetching one result: {e}")
            raise DatabaseError(f"Database operation failed: {e}")

    def fetch_all(self, query: str, params: tuple = ()) -> list:
        """Execute query and fetch all results."""
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error fetching all results: {e}")
            raise DatabaseError(f"Database operation failed: {e}")

@contextmanager
def get_db_connection() -> Generator[DatabaseConnection, None, None]:
    """Context manager for database connections."""
    conn = DatabaseConnection()
    try:
        with conn as db:
            yield db
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise DatabaseError(f"Failed to establish database connection: {e}")
