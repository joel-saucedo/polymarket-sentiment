"""
Unit tests for CLI functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import tempfile
import shutil
from pathlib import Path

from polymkt_sent.cli import cli, main


class TestCLI:
    """Test cases for CLI commands."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Polymarket Sentiment Analysis CLI' in result.output
    
    def test_cli_verbose_flag(self):
        """Test verbose flag."""
        result = self.runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0
    
    def test_cli_data_dir_option(self):
        """Test data directory option."""
        result = self.runner.invoke(cli, ['--data-dir', self.temp_dir, '--help'])
        assert result.exit_code == 0
    
    def test_scrape_help(self):
        """Test scrape command help."""
        result = self.runner.invoke(cli, ['scrape', '--help'])
        assert result.exit_code == 0
        assert 'Scraping commands' in result.output
    
    @patch('polymkt_sent.cli.TweetStorage')
    @patch('polymkt_sent.cli.AsyncBatchScraper')
    @patch('polymkt_sent.cli.asyncio.run')
    def test_scrape_run_with_users(self, mock_asyncio, mock_scraper_class, mock_storage_class):
        """Test scrape run command with users."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        
        mock_scraper = Mock()
        mock_scraper.scrape_all_targets.return_value = {
            'tweets_scraped': 10,
            'targets_processed': 2,
            'targets_total': 2,
            'duration_seconds': 5.0,
            'success_rate': 1.0
        }
        mock_scraper_class.return_value = mock_scraper
        mock_asyncio.return_value = mock_scraper.scrape_all_targets.return_value
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'scrape', 'run',
            '--users', 'testuser1',
            '--users', 'testuser2',
            '--max-tweets', '50'
        ])
        
        assert result.exit_code == 0
        assert 'Starting scraper with 2 targets' in result.output
        assert 'Scraping completed successfully' in result.output
        
        # Verify scraper was called with correct targets
        mock_asyncio.assert_called_once()
    
    @patch('polymkt_sent.cli.TweetStorage')
    def test_scrape_status(self, mock_storage_class):
        """Test scrape status command."""
        # Setup mock
        mock_storage = Mock()
        mock_storage.get_tweet_count.return_value = 100
        mock_storage.get_tweets.return_value = [
            {'username': 'user1', 'content': 'Test tweet content here'},
            {'username': 'user2', 'content': 'Another test tweet'}
        ]
        mock_storage_class.return_value = mock_storage
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'scrape', 'status'
        ])
        
        assert result.exit_code == 0
        assert 'Total tweets: 100' in result.output
        assert 'Recent tweets:' in result.output
    
    def test_sentiment_help(self):
        """Test sentiment command help."""
        result = self.runner.invoke(cli, ['sentiment', '--help'])
        assert result.exit_code == 0
        assert 'Sentiment analysis commands' in result.output
    
    @patch('polymkt_sent.cli.SentimentProcessor')
    @patch('polymkt_sent.cli.TweetStorage')
    def test_sentiment_analyze(self, mock_storage_class, mock_processor_class):
        """Test sentiment analyze command."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        
        mock_processor = Mock()
        mock_processor.process_batch.return_value = {
            'tweets_processed': 10,
            'tweets_updated': 10,
            'success_rate': 1.0,
            'tweets_per_second': 5.0
        }
        mock_processor_class.return_value = mock_processor
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'sentiment', 'analyze',
            '--batch-size', '50',
            '--vader-weight', '0.8',
            '--keyword-weight', '0.2'
        ])
        
        assert result.exit_code == 0
        assert 'Starting sentiment analysis' in result.output
        assert 'Tweets processed: 10' in result.output
        assert 'Success rate: 100.0%' in result.output
    
    @patch('polymkt_sent.cli.SentimentProcessor')
    @patch('polymkt_sent.cli.TweetStorage')
    def test_sentiment_summary(self, mock_storage_class, mock_processor_class):
        """Test sentiment summary command."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        
        mock_processor = Mock()
        mock_processor.get_sentiment_summary.return_value = {
            'total_tweets': 100,
            'positive_tweets': 60,
            'negative_tweets': 30,
            'neutral_tweets': 10,
            'average_sentiment': 0.25
        }
        mock_processor_class.return_value = mock_processor
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'sentiment', 'summary'
        ])
        
        assert result.exit_code == 0
        assert 'Total tweets: 100' in result.output
        assert 'Positive: 60' in result.output
        assert 'Average sentiment: 0.250' in result.output
    
    def test_api_help(self):
        """Test API command help."""
        result = self.runner.invoke(cli, ['api', '--help'])
        assert result.exit_code == 0
        assert 'API server commands' in result.output
    
    @patch('polymkt_sent.cli.run_server')
    def test_api_serve(self, mock_run_server):
        """Test API serve command."""
        # Mock server to avoid actually starting it
        mock_run_server.return_value = None
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'api', 'serve',
            '--host', '127.0.0.1',
            '--port', '8001'
        ])
        
        assert result.exit_code == 0
        mock_run_server.assert_called_once_with(
            host='127.0.0.1',
            port=8001,
            reload=False
        )
    
    def test_data_help(self):
        """Test data command help."""
        result = self.runner.invoke(cli, ['data', '--help'])
        assert result.exit_code == 0
        assert 'Data management commands' in result.output
    
    @patch('polymkt_sent.cli.TweetStorage')
    def test_data_clear_with_confirmation(self, mock_storage_class):
        """Test data clear command with confirmation."""
        mock_storage = Mock()
        mock_storage.get_tweet_count.return_value = 50
        mock_storage_class.return_value = mock_storage
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'data', 'clear',
            '--confirm'
        ])
        
        assert result.exit_code == 0
        assert 'Would clear 50 tweets' in result.output
    
    @patch('polymkt_sent.cli.TweetStorage')
    def test_data_export(self, mock_storage_class):
        """Test data export command."""
        mock_storage = Mock()
        mock_storage.get_tweets.return_value = [
            {'tweet_id': '1', 'content': 'Test tweet'},
            {'tweet_id': '2', 'content': 'Another tweet'}
        ]
        mock_storage_class.return_value = mock_storage
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'data', 'export',
            '--format', 'json',
            '--with-sentiment'
        ])
        
        assert result.exit_code == 0
        assert 'Would export 2 tweets' in result.output
    
    @patch('polymkt_sent.cli.SentimentProcessor')
    @patch('polymkt_sent.cli.AsyncBatchScraper')
    @patch('polymkt_sent.cli.TweetStorage')
    @patch('polymkt_sent.cli.asyncio.run')
    def test_run_pipeline(self, mock_asyncio, mock_storage_class, mock_scraper_class, mock_processor_class):
        """Test complete pipeline command."""
        # Setup mocks
        mock_storage = Mock()
        mock_storage_class.return_value = mock_storage
        
        mock_scraper = Mock()
        mock_scraper.scrape_all_targets.return_value = {
            'tweets_scraped': 5,
            'targets_processed': 2,
            'targets_total': 2,
            'duration_seconds': 3.0,
            'success_rate': 1.0
        }
        mock_scraper_class.return_value = mock_scraper
        
        mock_processor = Mock()
        mock_processor.process_batch.return_value = {
            'tweets_processed': 5,
            'tweets_updated': 5,
            'success_rate': 1.0,
            'tweets_per_second': 2.0
        }
        mock_processor_class.return_value = mock_processor
        
        mock_asyncio.return_value = mock_scraper.scrape_all_targets.return_value
        
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'run-pipeline',
            '--users', 'testuser',
            '--terms', 'crypto',
            '--analyze-sentiment'
        ])
        
        assert result.exit_code == 0
        assert 'Starting polymarket-sentiment pipeline' in result.output
        assert 'Pipeline completed successfully' in result.output


class TestCLIMain:
    """Test cases for CLI main function."""
    
    @patch('polymkt_sent.cli.cli')
    def test_main_function(self, mock_cli):
        """Test main function calls CLI."""
        mock_cli.return_value = None
        
        result = main()
        mock_cli.assert_called_once()
    
    @patch('polymkt_sent.cli.cli')
    def test_main_keyboard_interrupt(self, mock_cli):
        """Test main function handles KeyboardInterrupt."""
        mock_cli.side_effect = KeyboardInterrupt()
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 130
    
    @patch('polymkt_sent.cli.cli')
    def test_main_exception_handling(self, mock_cli):
        """Test main function handles exceptions."""
        mock_cli.side_effect = Exception("Test error")
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1


class TestCLIIntegration:
    """Integration tests for CLI with real components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        Path(self.temp_dir).mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_directory_creation(self):
        """Test that CLI creates data directory."""
        test_dir = Path(self.temp_dir) / "test_data"
        
        # Use scrape status command which should execute the main callback
        result = self.runner.invoke(cli, [
            '--data-dir', str(test_dir),
            'scrape', 'status'
        ])
        
        # Command might fail but directory should be created
        assert test_dir.exists()
    
    def test_cli_without_data(self):
        """Test CLI commands work with empty data directory."""
        # This should not crash even with no data
        result = self.runner.invoke(cli, [
            '--data-dir', self.temp_dir,
            'scrape', 'status'
        ])
        
        # Should handle gracefully (may show error but shouldn't crash)
        assert result.exit_code in [0, 1]  # Either success or handled error
