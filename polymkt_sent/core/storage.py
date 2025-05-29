"""
Tweet storage layer using DuckDB and Parquet for polymarket-sentiment.

This module provides:
- Tweet data modeling with Pydantic
- DuckDB integration for fast querying
- Parquet export for data persistence
- Time-series indexing for efficient retrieval
- Sentiment score storage and aggregation
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import pandas as pd
import duckdb
from pydantic import BaseModel, Field, field_validator, ConfigDict
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class TweetModel(BaseModel):
    """Pydantic model for tweet data with validation."""
    
    tweet_id: str = Field(..., description="Unique tweet identifier")
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username without @")
    content: str = Field(..., description="Tweet text content")
    timestamp: datetime = Field(..., description="Tweet creation timestamp")
    scraped_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Engagement metrics
    likes: int = Field(default=0, ge=0)
    retweets: int = Field(default=0, ge=0)
    replies: int = Field(default=0, ge=0)
    
    # Metadata
    is_reply: bool = Field(default=False)
    is_retweet: bool = Field(default=False)
    reply_to: Optional[str] = Field(default=None)
    source_instance: Optional[str] = Field(default=None)
    
    # Sentiment data (populated later)
    sentiment_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    sentiment_label: Optional[str] = Field(default=None)
    sentiment_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    @field_validator('timestamp', 'scraped_at')
    @classmethod
    def ensure_timezone(cls, v):
        """Ensure timestamps have timezone info."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate tweet content."""
        if not v or not v.strip():
            raise ValueError("Tweet content cannot be empty")
        return v.strip()
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


@dataclass
class StorageConfig:
    """Configuration for storage layer."""
    
    data_dir: str = "data"
    db_file: str = "tweets.duckdb"
    parquet_file: str = "tweets.parquet"
    batch_size: int = 1000
    auto_export: bool = True
    retention_days: Optional[int] = None
    
    def __post_init__(self):
        """Ensure data directory exists."""
        Path(self.data_dir).mkdir(exist_ok=True)
    
    @property
    def db_path(self) -> str:
        """Full path to DuckDB file."""
        return os.path.join(self.data_dir, self.db_file)
    
    @property 
    def parquet_path(self) -> str:
        """Full path to Parquet file."""
        return os.path.join(self.data_dir, self.parquet_file)


class TweetStorage:
    """Tweet storage manager using DuckDB and Parquet."""
    
    def __init__(self, config: StorageConfig):
        """Initialize storage with configuration."""
        self.config = config
        self.connection = None
        self._setup_database()
        
    def _setup_database(self) -> None:
        """Setup DuckDB connection and create tables."""
        try:
            self.connection = duckdb.connect(self.config.db_path)
            self._create_tables()
            self._create_indexes()
            logger.info(f"Database initialized at {self.config.db_path}")
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    def _create_tables(self) -> None:
        """Create necessary tables."""
        tweets_table = """
        CREATE TABLE IF NOT EXISTS tweets (
            tweet_id VARCHAR PRIMARY KEY,
            user_id VARCHAR NOT NULL,
            username VARCHAR NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            scraped_at TIMESTAMPTZ NOT NULL,
            likes INTEGER DEFAULT 0,
            retweets INTEGER DEFAULT 0,
            replies INTEGER DEFAULT 0,
            is_reply BOOLEAN DEFAULT FALSE,
            is_retweet BOOLEAN DEFAULT FALSE,
            reply_to VARCHAR,
            source_instance VARCHAR,
            sentiment_score DOUBLE,
            sentiment_label VARCHAR,
            sentiment_confidence DOUBLE
        )
        """
        
        # Table for tracking data exports
        exports_table = """
        CREATE TABLE IF NOT EXISTS exports (
            export_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            export_type VARCHAR NOT NULL,
            file_path VARCHAR NOT NULL,
            row_count INTEGER NOT NULL,
            exported_at TIMESTAMPTZ DEFAULT now(),
            checksum VARCHAR
        )
        """
        
        self.connection.execute(tweets_table)
        self.connection.execute(exports_table)
        
    def _create_indexes(self) -> None:
        """Create indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tweets_username ON tweets(username)",
            "CREATE INDEX IF NOT EXISTS idx_tweets_scraped_at ON tweets(scraped_at)",
            "CREATE INDEX IF NOT EXISTS idx_tweets_sentiment ON tweets(sentiment_score) WHERE sentiment_score IS NOT NULL",
        ]
        
        for index in indexes:
            try:
                self.connection.execute(index)
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")
    
    def insert_tweet(self, tweet: TweetModel) -> bool:
        """Insert a single tweet."""
        try:
            self.connection.execute("""
                INSERT OR REPLACE INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                tweet.tweet_id, tweet.user_id, tweet.username, tweet.content,
                tweet.timestamp, tweet.scraped_at, tweet.likes, tweet.retweets,
                tweet.replies, tweet.is_reply, tweet.is_retweet, tweet.reply_to,
                tweet.source_instance, tweet.sentiment_score, tweet.sentiment_label,
                tweet.sentiment_confidence
            ])
            return True
        except Exception as e:
            logger.error(f"Failed to insert tweet {tweet.tweet_id}: {e}")
            return False
    
    def insert_tweets_batch(self, tweets: List[TweetModel]) -> int:
        """Insert multiple tweets in a batch."""
        if not tweets:
            return 0
            
        successful = 0
        try:
            # Prepare data for batch insert
            data = []
            for tweet in tweets:
                data.append([
                    tweet.tweet_id, tweet.user_id, tweet.username, tweet.content,
                    tweet.timestamp, tweet.scraped_at, tweet.likes, tweet.retweets,
                    tweet.replies, tweet.is_reply, tweet.is_retweet, tweet.reply_to,
                    tweet.source_instance, tweet.sentiment_score, tweet.sentiment_label,
                    tweet.sentiment_confidence
                ])
            
            # Use executemany for better performance
            self.connection.executemany("""
                INSERT OR REPLACE INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            
            successful = len(tweets)
            logger.info(f"Successfully inserted {successful} tweets")
            
            # Auto-export if enabled
            if self.config.auto_export:
                self.export_to_parquet()
                
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            # Try individual inserts as fallback
            for tweet in tweets:
                if self.insert_tweet(tweet):
                    successful += 1
        
        return successful
    
    def get_tweets(
        self,
        username: Optional[str] = None,
        tweet_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        with_sentiment: bool = False
    ) -> List[Dict[str, Any]]:
        """Retrieve tweets with optional filtering."""
        query = "SELECT * FROM tweets WHERE 1=1"
        params = []
        
        if username:
            query += " AND username = ?"
            params.append(username)
        
        if tweet_id:
            query += " AND tweet_id = ?"
            params.append(tweet_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if with_sentiment:
            query += " AND sentiment_score IS NOT NULL"
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            result = self.connection.execute(query, params).fetchall()
            columns = [desc[0] for desc in self.connection.description]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Failed to retrieve tweets: {e}")
            return []
    
    def get_sentiment_summary(
        self,
        username: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get sentiment summary for recent tweets."""
        query = """
        SELECT 
            COUNT(*) as total_tweets,
            AVG(sentiment_score) as avg_sentiment,
            COUNT(CASE WHEN sentiment_score > 0.1 THEN 1 END) as positive_tweets,
            COUNT(CASE WHEN sentiment_score < -0.1 THEN 1 END) as negative_tweets,
            COUNT(CASE WHEN sentiment_score BETWEEN -0.1 AND 0.1 THEN 1 END) as neutral_tweets,
            MIN(timestamp) as earliest_tweet,
            MAX(timestamp) as latest_tweet
        FROM tweets 
        WHERE sentiment_score IS NOT NULL 
        AND timestamp >= now() - INTERVAL '{hours} hours'
        """.format(hours=hours)
        
        params = []
        if username:
            query += " AND username = ?"
            params.append(username)
        
        try:
            result = self.connection.execute(query, params).fetchone()
            if result:
                columns = [desc[0] for desc in self.connection.description]
                return dict(zip(columns, result))
            return {}
        except Exception as e:
            logger.error(f"Failed to get sentiment summary: {e}")
            return {}
    
    def update_sentiment(
        self,
        tweet_id: str,
        sentiment_score: float,
        sentiment_label: str,
        confidence: float
    ) -> bool:
        """Update sentiment data for a tweet."""
        try:
            self.connection.execute("""
                UPDATE tweets 
                SET sentiment_score = ?, sentiment_label = ?, sentiment_confidence = ?
                WHERE tweet_id = ?
            """, [sentiment_score, sentiment_label, confidence, tweet_id])
            return True
        except Exception as e:
            logger.error(f"Failed to update sentiment for {tweet_id}: {e}")
            return False
    
    def export_to_parquet(self) -> bool:
        """Export all tweets to Parquet file."""
        try:
            # Export tweets to Parquet
            self.connection.execute(f"""
                COPY (SELECT * FROM tweets ORDER BY timestamp DESC) 
                TO '{self.config.parquet_path}' (FORMAT PARQUET)
            """)
            
            # Record export metadata
            row_count = self.connection.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
            self.connection.execute("""
                INSERT INTO exports (export_type, file_path, row_count) 
                VALUES (?, ?, ?)
            """, ["parquet", self.config.parquet_path, row_count])
            
            logger.info(f"Exported {row_count} tweets to {self.config.parquet_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export to Parquet: {e}")
            return False
    
    def cleanup_old_data(self, days: int) -> int:
        """Remove tweets older than specified days."""
        try:
            # First count the rows to be deleted
            count_result = self.connection.execute("""
                SELECT COUNT(*) FROM tweets 
                WHERE timestamp < now() - INTERVAL '{days} days'
            """.format(days=days)).fetchone()
            
            deleted_count = count_result[0] if count_result else 0
            
            # Then delete the rows
            self.connection.execute("""
                DELETE FROM tweets 
                WHERE timestamp < now() - INTERVAL '{days} days'
            """.format(days=days))
            
            logger.info(f"Cleaned up {deleted_count} old tweets")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Basic counts
            result = self.connection.execute("""
                SELECT 
                    COUNT(*) as total_tweets,
                    COUNT(DISTINCT username) as unique_users,
                    COUNT(CASE WHEN sentiment_score IS NOT NULL THEN 1 END) as tweets_with_sentiment,
                    MIN(timestamp) as earliest_tweet,
                    MAX(timestamp) as latest_tweet,
                    MAX(scraped_at) as last_scraped
                FROM tweets
            """).fetchone()
            
            if result:
                columns = [desc[0] for desc in self.connection.description]
                stats.update(dict(zip(columns, result)))
            
            # File sizes
            if os.path.exists(self.config.db_path):
                stats['db_size_mb'] = os.path.getsize(self.config.db_path) / (1024 * 1024)
            
            if os.path.exists(self.config.parquet_path):
                stats['parquet_size_mb'] = os.path.getsize(self.config.parquet_path) / (1024 * 1024)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


def create_storage_from_config(config: Dict[str, Any]) -> TweetStorage:
    """Create storage instance from configuration dictionary."""
    storage_config_dict = config.get("storage", {})
    
    storage_config = StorageConfig(
        data_dir=storage_config_dict.get("data_dir", "data"),
        db_file=storage_config_dict.get("db_file", "tweets.duckdb"),
        parquet_file=storage_config_dict.get("parquet_file", "tweets.parquet"),
        batch_size=storage_config_dict.get("batch_size", 1000),
        auto_export=storage_config_dict.get("auto_export", True),
        retention_days=storage_config_dict.get("retention_days")
    )
    
    return TweetStorage(storage_config)
