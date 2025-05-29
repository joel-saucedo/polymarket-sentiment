"""
Unit tests for polymkt_sent.core.storage module.
"""

import pytest
import tempfile
import shutil
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, Mock

from polymkt_sent.core.storage import (
    TweetModel,
    StorageConfig,
    TweetStorage,
    create_storage_from_config
)


class TestTweetModel:
    """Test cases for TweetModel."""
    
    def test_tweet_model_creation(self):
        """Test creating a valid tweet model."""
        now = datetime.now(timezone.utc)
        
        tweet = TweetModel(
            tweet_id="123456789",
            user_id="user123",
            username="testuser",
            content="This is a test tweet",
            timestamp=now
        )
        
        assert tweet.tweet_id == "123456789"
        assert tweet.username == "testuser"
        assert tweet.content == "This is a test tweet"
        assert tweet.timestamp == now
        assert tweet.scraped_at <= datetime.now(timezone.utc)
        assert tweet.likes == 0
        assert tweet.sentiment_score is None
    
    def test_tweet_model_with_metrics(self):
        """Test tweet model with engagement metrics."""
        tweet = TweetModel(
            tweet_id="123",
            user_id="user123",
            username="testuser",
            content="Popular tweet",
            timestamp=datetime.now(timezone.utc),
            likes=100,
            retweets=50,
            replies=25
        )
        
        assert tweet.likes == 100
        assert tweet.retweets == 50
        assert tweet.replies == 25
    
    def test_tweet_model_with_sentiment(self):
        """Test tweet model with sentiment data."""
        tweet = TweetModel(
            tweet_id="123",
            user_id="user123",
            username="testuser",
            content="Great news!",
            timestamp=datetime.now(timezone.utc),
            sentiment_score=0.8,
            sentiment_label="positive",
            sentiment_confidence=0.9
        )
        
        assert tweet.sentiment_score == 0.8
        assert tweet.sentiment_label == "positive"
        assert tweet.sentiment_confidence == 0.9
    
    def test_tweet_model_timezone_handling(self):
        """Test timezone handling in timestamps."""
        naive_time = datetime(2025, 5, 29, 12, 0, 0)
        
        tweet = TweetModel(
            tweet_id="123",
            user_id="user123",
            username="testuser",
            content="Test",
            timestamp=naive_time
        )
        
        # Should add UTC timezone to naive datetime
        assert tweet.timestamp.tzinfo == timezone.utc
    
    def test_tweet_model_content_validation(self):
        """Test content validation."""
        with pytest.raises(ValueError, match="Tweet content cannot be empty"):
            TweetModel(
                tweet_id="123",
                user_id="user123",
                username="testuser",
                content="",
                timestamp=datetime.now(timezone.utc)
            )
    
    def test_tweet_model_sentiment_validation(self):
        """Test sentiment score validation."""
        with pytest.raises(ValueError):
            TweetModel(
                tweet_id="123",
                user_id="user123",
                username="testuser",
                content="Test",
                timestamp=datetime.now(timezone.utc),
                sentiment_score=2.0  # Invalid: > 1.0
            )
    
    def test_tweet_model_negative_metrics(self):
        """Test validation of negative engagement metrics."""
        with pytest.raises(ValueError):
            TweetModel(
                tweet_id="123",
                user_id="user123",
                username="testuser",
                content="Test",
                timestamp=datetime.now(timezone.utc),
                likes=-1  # Invalid: negative
            )


class TestStorageConfig:
    """Test cases for StorageConfig."""
    
    def test_default_config(self):
        """Test default storage configuration."""
        config = StorageConfig()
        
        assert config.data_dir == "data"
        assert config.db_file == "tweets.duckdb"
        assert config.parquet_file == "tweets.parquet"
        assert config.batch_size == 1000
        assert config.auto_export is True
        assert config.retention_days is None
    
    def test_custom_config(self):
        """Test custom storage configuration."""
        config = StorageConfig(
            data_dir="/tmp/test_data",
            db_file="custom.duckdb",
            batch_size=500,
            retention_days=30
        )
        
        assert config.data_dir == "/tmp/test_data"
        assert config.db_file == "custom.duckdb"
        assert config.batch_size == 500
        assert config.retention_days == 30
    
    def test_path_properties(self):
        """Test path property methods."""
        config = StorageConfig(
            data_dir="/tmp/test",
            db_file="test.duckdb",
            parquet_file="test.parquet"
        )
        
        assert config.db_path == "/tmp/test/test.duckdb"
        assert config.parquet_path == "/tmp/test/test.parquet"
    
    @patch('pathlib.Path.mkdir')
    def test_directory_creation(self, mock_mkdir):
        """Test that data directory is created."""
        StorageConfig(data_dir="/tmp/new_dir")
        mock_mkdir.assert_called_once_with(exist_ok=True)


class TestTweetStorage:
    """Test cases for TweetStorage."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = StorageConfig(
            data_dir=self.temp_dir,
            db_file="test.duckdb",
            parquet_file="test.parquet",
            auto_export=False  # Disable for tests
        )
        self.storage = TweetStorage(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'storage'):
            self.storage.close()
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_storage_initialization(self):
        """Test storage initialization."""
        assert self.storage.config == self.config
        assert self.storage.connection is not None
        assert os.path.exists(self.config.db_path)
    
    def test_insert_single_tweet(self):
        """Test inserting a single tweet."""
        tweet = TweetModel(
            tweet_id="123",
            user_id="user123",
            username="testuser",
            content="Test tweet",
            timestamp=datetime.now(timezone.utc)
        )
        
        result = self.storage.insert_tweet(tweet)
        assert result is True
        
        # Verify insertion
        tweets = self.storage.get_tweets(limit=1)
        assert len(tweets) == 1
        assert tweets[0]['tweet_id'] == "123"
        assert tweets[0]['username'] == "testuser"
    
    def test_insert_batch_tweets(self):
        """Test batch insert of tweets."""
        tweets = []
        for i in range(5):
            tweets.append(TweetModel(
                tweet_id=f"tweet_{i}",
                user_id=f"user_{i}",
                username=f"user{i}",
                content=f"Test tweet {i}",
                timestamp=datetime.now(timezone.utc)
            ))
        
        result = self.storage.insert_tweets_batch(tweets)
        assert result == 5
        
        # Verify insertion
        stored_tweets = self.storage.get_tweets()
        assert len(stored_tweets) == 5
    
    def test_insert_empty_batch(self):
        """Test inserting empty batch."""
        result = self.storage.insert_tweets_batch([])
        assert result == 0
    
    def test_get_tweets_with_filters(self):
        """Test retrieving tweets with filters."""
        # Insert test tweets
        base_time = datetime.now(timezone.utc)
        tweets = [
            TweetModel(
                tweet_id="1",
                user_id="user1",
                username="alice",
                content="Alice tweet 1",
                timestamp=base_time - timedelta(hours=2)
            ),
            TweetModel(
                tweet_id="2",
                user_id="user1",
                username="alice",
                content="Alice tweet 2",
                timestamp=base_time - timedelta(hours=1)
            ),
            TweetModel(
                tweet_id="3",
                user_id="user2",
                username="bob",
                content="Bob tweet",
                timestamp=base_time
            )
        ]
        
        self.storage.insert_tweets_batch(tweets)
        
        # Test username filter
        alice_tweets = self.storage.get_tweets(username="alice")
        assert len(alice_tweets) == 2
        assert all(t['username'] == 'alice' for t in alice_tweets)
        
        # Test time filter
        recent_tweets = self.storage.get_tweets(
            start_time=base_time - timedelta(minutes=30)
        )
        assert len(recent_tweets) == 1
        assert recent_tweets[0]['username'] == 'bob'
        
        # Test limit
        limited_tweets = self.storage.get_tweets(limit=1)
        assert len(limited_tweets) == 1
    
    def test_update_sentiment(self):
        """Test updating sentiment for a tweet."""
        # Insert test tweet
        tweet = TweetModel(
            tweet_id="123",
            user_id="user123",
            username="testuser",
            content="Great news!",
            timestamp=datetime.now(timezone.utc)
        )
        self.storage.insert_tweet(tweet)
        
        # Update sentiment
        result = self.storage.update_sentiment("123", 0.8, "positive", 0.9)
        assert result is True
        
        # Verify update
        tweets = self.storage.get_tweets(tweet_id="123")
        # Note: Need to modify get_tweets to support tweet_id filter
        # For now, get all and filter
        all_tweets = self.storage.get_tweets()
        tweet_data = next(t for t in all_tweets if t['tweet_id'] == "123")
        assert tweet_data['sentiment_score'] == 0.8
        assert tweet_data['sentiment_label'] == "positive"
        assert tweet_data['sentiment_confidence'] == 0.9
    
    def test_get_sentiment_summary(self):
        """Test sentiment summary calculation."""
        # Insert tweets with sentiment
        tweets = [
            TweetModel(
                tweet_id="1",
                user_id="user1",
                username="testuser",
                content="Positive tweet",
                timestamp=datetime.now(timezone.utc),
                sentiment_score=0.5
            ),
            TweetModel(
                tweet_id="2",
                user_id="user1",
                username="testuser",
                content="Negative tweet",
                timestamp=datetime.now(timezone.utc),
                sentiment_score=-0.5
            ),
            TweetModel(
                tweet_id="3",
                user_id="user1",
                username="testuser",
                content="Neutral tweet",
                timestamp=datetime.now(timezone.utc),
                sentiment_score=0.0
            )
        ]
        
        self.storage.insert_tweets_batch(tweets)
        
        summary = self.storage.get_sentiment_summary()
        assert summary['total_tweets'] == 3
        assert summary['positive_tweets'] == 1
        assert summary['negative_tweets'] == 1
        assert summary['neutral_tweets'] == 1
        assert summary['avg_sentiment'] == 0.0  # (0.5 + (-0.5) + 0.0) / 3
    
    def test_get_stats(self):
        """Test database statistics."""
        # Insert some test data
        tweets = [
            TweetModel(
                tweet_id=f"tweet_{i}",
                user_id=f"user_{i % 2}",  # 2 unique users
                username=f"user{i % 2}",
                content=f"Tweet {i}",
                timestamp=datetime.now(timezone.utc),
                sentiment_score=0.1 if i % 2 == 0 else None
            )
            for i in range(5)
        ]
        
        self.storage.insert_tweets_batch(tweets)
        
        stats = self.storage.get_stats()
        assert stats['total_tweets'] == 5
        assert stats['unique_users'] == 2
        assert stats['tweets_with_sentiment'] == 3  # 0, 2, 4 have sentiment
        assert 'db_size_mb' in stats
    
    @patch('duckdb.connect')
    def test_database_connection_error(self, mock_connect):
        """Test handling of database connection errors."""
        mock_connect.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            TweetStorage(self.config)
    
    def test_export_to_parquet(self):
        """Test Parquet export functionality."""
        # Insert test data
        tweet = TweetModel(
            tweet_id="123",
            user_id="user123",
            username="testuser",
            content="Test tweet",
            timestamp=datetime.now(timezone.utc)
        )
        self.storage.insert_tweet(tweet)
        
        # Export to Parquet
        result = self.storage.export_to_parquet()
        assert result is True
        assert os.path.exists(self.config.parquet_path)
    
    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        # Insert old and new tweets
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        new_time = datetime.now(timezone.utc)
        
        tweets = [
            TweetModel(
                tweet_id="old",
                user_id="user1",
                username="testuser",
                content="Old tweet",
                timestamp=old_time
            ),
            TweetModel(
                tweet_id="new",
                user_id="user1",
                username="testuser",
                content="New tweet",
                timestamp=new_time
            )
        ]
        
        self.storage.insert_tweets_batch(tweets)
        
        # Cleanup data older than 5 days
        deleted_count = self.storage.cleanup_old_data(5)
        assert deleted_count == 1
        
        # Verify only new tweet remains
        remaining_tweets = self.storage.get_tweets()
        assert len(remaining_tweets) == 1
        assert remaining_tweets[0]['tweet_id'] == "new"


class TestCreateStorageFromConfig:
    """Test cases for create_storage_from_config function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_with_default_config(self):
        """Test creating storage with default config."""
        config = {}
        storage = create_storage_from_config(config)
        
        assert storage.config.data_dir == "data"
        assert storage.config.db_file == "tweets.duckdb"
        storage.close()
    
    def test_create_with_custom_config(self):
        """Test creating storage with custom config."""
        config = {
            "storage": {
                "data_dir": self.temp_dir,
                "db_file": "custom.duckdb",
                "batch_size": 500,
                "auto_export": False,
                "retention_days": 7
            }
        }
        
        storage = create_storage_from_config(config)
        
        assert storage.config.data_dir == self.temp_dir
        assert storage.config.db_file == "custom.duckdb"
        assert storage.config.batch_size == 500
        assert storage.config.auto_export is False
        assert storage.config.retention_days == 7
        storage.close()
    
    def test_create_with_partial_config(self):
        """Test creating storage with partial config."""
        config = {
            "storage": {
                "data_dir": self.temp_dir,
                "batch_size": 2000
            }
        }
        
        storage = create_storage_from_config(config)
        
        assert storage.config.data_dir == self.temp_dir
        assert storage.config.batch_size == 2000
        assert storage.config.db_file == "tweets.duckdb"  # default
        storage.close()


if __name__ == "__main__":
    pytest.main([__file__])
