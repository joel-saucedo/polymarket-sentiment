"""
Command-line interface for polymarket-sentiment.

Provides commands for:
- Running the scraper
- Analyzing sentiment
- Starting the API server
- Managing data
"""

import asyncio
import logging
import sys
from pathlib import Path
import click
from typing import Optional

from polymkt_sent.core.storage import TweetStorage, StorageConfig
from polymkt_sent.core.scraper import AsyncBatchScraper, ScrapingConfig, load_scraping_config, load_scraping_targets
from polymkt_sent.core.sentiment import SentimentProcessor, SentimentConfig, create_sentiment_processor
from polymkt_sent.api import run_server


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Main CLI group
@click.group()
@click.option('--data-dir', default='data', help='Data directory path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, data_dir: str, verbose: bool):
    """Polymarket Sentiment Analysis CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir
    
    # Create data directory if it doesn't exist (even for help commands)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)


# Scraper commands
@cli.group()
def scrape():
    """Scraping commands."""
    pass


@scrape.command()
@click.option('--users', '-u', multiple=True, help='Twitter usernames to scrape')
@click.option('--terms', '-t', multiple=True, help='Search terms to scrape')
@click.option('--max-tweets', default=100, help='Maximum tweets per target')
@click.option('--config-file', help='Path to scraper config file')
@click.pass_context
def run(ctx, users, terms, max_tweets: int, config_file: Optional[str]):
    """Run the tweet scraper."""
    data_dir = ctx.obj['data_dir']
    
    try:
        # Create storage
        storage_config = StorageConfig(data_dir=data_dir)
        storage = TweetStorage(storage_config)
        
        # Create scraper config
        if config_file and Path(config_file).exists():
            scraper_config = load_scraping_config(config_file)
        else:
            scraper_config = ScrapingConfig()
        
        # Override tweets per target if specified
        scraper_config.tweets_per_target = max_tweets
        
        # Prepare targets
        from polymkt_sent.core.scraper import ScrapingTarget
        targets = []
        
        for user in users:
            targets.append(ScrapingTarget(
                target_type="user", 
                value=user, 
                description=f"User account: {user}"
            ))
        
        for term in terms:
            targets.append(ScrapingTarget(
                target_type="search", 
                value=term, 
                description=f"Search term: {term}"
            ))
        
        # Load targets from config if no CLI targets specified
        if not targets and config_file:
            targets_config_path = Path(config_file).parent / "accounts.yml"
            if targets_config_path.exists():
                targets = load_scraping_targets(str(targets_config_path))
        
        if not targets:
            click.echo("No targets specified. Use --users or --terms to specify what to scrape.")
            return
        
        click.echo(f"Starting scraper with {len(targets)} targets...")
        
        # Create and run scraper
        scraper = AsyncBatchScraper(storage, scraper_config, targets)
        summary = asyncio.run(scraper.scrape_all_targets())
        
        # Display results
        click.echo(f"Scraping completed successfully!")
        click.echo(f"  • Scraped {summary['tweets_scraped']} tweets")
        click.echo(f"  • Processed {summary['targets_processed']}/{summary['targets_total']} targets")
        click.echo(f"  • Duration: {summary['duration_seconds']:.1f} seconds")
        click.echo(f"  • Success rate: {summary['success_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@scrape.command()
@click.pass_context
def status(ctx):
    """Show scraper status and statistics."""
    data_dir = ctx.obj['data_dir']
    
    try:
        storage_config = StorageConfig(data_dir=data_dir)
        storage = TweetStorage(storage_config)
        
        total_tweets = storage.get_tweet_count()
        tweets_with_sentiment = len(storage.get_tweets(with_sentiment=True, limit=1000))
        
        click.echo("Scraper Status:")
        click.echo(f"  Total tweets: {total_tweets}")
        click.echo(f"  Tweets with sentiment: {tweets_with_sentiment}")
        click.echo(f"  Data directory: {data_dir}")
        
        # Show recent activity
        recent_tweets = storage.get_tweets(limit=5)
        if recent_tweets:
            click.echo("\nRecent tweets:")
            for tweet in recent_tweets[:3]:
                click.echo(f"  - @{tweet['username']}: {tweet['content'][:50]}...")
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Sentiment commands
@cli.group()
def sentiment():
    """Sentiment analysis commands."""
    pass


@sentiment.command()
@click.option('--batch-size', default=100, help='Batch size for processing')
@click.option('--vader-weight', default=0.7, help='VADER analyzer weight')
@click.option('--keyword-weight', default=0.3, help='Keyword analyzer weight')
@click.pass_context
def analyze(ctx, batch_size: int, vader_weight: float, keyword_weight: float):
    """Analyze sentiment for all tweets."""
    data_dir = ctx.obj['data_dir']
    
    try:
        # Create storage and sentiment processor
        storage_config = StorageConfig(data_dir=data_dir)
        storage = TweetStorage(storage_config)
        
        sentiment_config = SentimentConfig(
            vader_weight=vader_weight,
            keyword_weight=keyword_weight
        )
        processor = SentimentProcessor(storage, sentiment_config)
        
        click.echo("Starting sentiment analysis...")
        
        # Process sentiment
        summary = processor.process_batch(batch_size=batch_size)
        
        click.echo("Sentiment Analysis Complete!")
        click.echo(f"  Tweets processed: {summary['tweets_processed']}")
        click.echo(f"  Tweets updated: {summary['tweets_updated']}")
        click.echo(f"  Success rate: {summary['success_rate']:.1%}")
        click.echo(f"  Processing speed: {summary['tweets_per_second']:.1f} tweets/sec")
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@sentiment.command()
@click.pass_context
def summary(ctx):
    """Show sentiment analysis summary."""
    data_dir = ctx.obj['data_dir']
    
    try:
        storage_config = StorageConfig(data_dir=data_dir)
        storage = TweetStorage(storage_config)
        
        sentiment_config = SentimentConfig()
        processor = SentimentProcessor(storage, sentiment_config)
        
        summary = processor.get_sentiment_summary()
        
        if "error" in summary:
            click.echo(f"Error: {summary['error']}", err=True)
            return
        
        click.echo("Sentiment Summary:")
        click.echo(f"  Total tweets: {summary.get('total_tweets', 0)}")
        click.echo(f"  Positive: {summary.get('positive_tweets', 0)}")
        click.echo(f"  Negative: {summary.get('negative_tweets', 0)}")
        click.echo(f"  Neutral: {summary.get('neutral_tweets', 0)}")
        click.echo(f"  Average sentiment: {summary.get('average_sentiment', 0.0):.3f}")
        
    except Exception as e:
        logger.error(f"Failed to get sentiment summary: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# API commands
@cli.group()
def api():
    """API server commands."""
    pass


@api.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
@click.pass_context
def serve(ctx, host: str, port: int, reload: bool):
    """Start the API server."""
    data_dir = ctx.obj['data_dir']
    
    click.echo(f"Starting API server on {host}:{port}")
    click.echo(f"Data directory: {data_dir}")
    
    try:
        run_server(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        click.echo("\nShutting down server...")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Data management commands
@cli.group()
def data():
    """Data management commands."""
    pass


@data.command()
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def clear(ctx, confirm: bool):
    """Clear all stored data."""
    data_dir = ctx.obj['data_dir']
    
    if not confirm:
        if not click.confirm(f"This will delete all data in {data_dir}. Continue?"):
            click.echo("Cancelled.")
            return
    
    try:
        storage_config = StorageConfig(data_dir=data_dir)
        storage = TweetStorage(storage_config)
        
        # Get count before clearing
        count = storage.get_tweet_count()
        
        # Clear data (implement this method in storage)
        # For now, just show what would be cleared
        click.echo(f"Would clear {count} tweets from {data_dir}")
        click.echo("Note: Clear functionality needs to be implemented in TweetStorage")
        
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--format', 'output_format', default='json', 
              type=click.Choice(['json', 'csv', 'parquet']), 
              help='Export format')
@click.option('--output', '-o', help='Output file path')
@click.option('--with-sentiment', is_flag=True, help='Include sentiment data')
@click.pass_context
def export(ctx, output_format: str, output: Optional[str], with_sentiment: bool):
    """Export data to file."""
    data_dir = ctx.obj['data_dir']
    
    try:
        storage_config = StorageConfig(data_dir=data_dir)
        storage = TweetStorage(storage_config)
        
        # Get tweets
        tweets = storage.get_tweets(with_sentiment=with_sentiment)
        
        if not tweets:
            click.echo("No tweets to export.")
            return
        
        # Determine output file
        if not output:
            output = f"tweets_export.{output_format}"
        
        # Export data (implement export methods in storage)
        click.echo(f"Would export {len(tweets)} tweets to {output} in {output_format} format")
        click.echo("Note: Export functionality needs to be implemented in TweetStorage")
        
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Pipeline command
@cli.command()
@click.option('--users', '-u', multiple=True, help='Twitter usernames to scrape')
@click.option('--terms', '-t', multiple=True, help='Search terms to scrape')
@click.option('--max-tweets', default=100, help='Maximum tweets per target')
@click.option('--analyze-sentiment', is_flag=True, help='Run sentiment analysis after scraping')
@click.pass_context
def run_pipeline(ctx, users, terms, max_tweets: int, analyze_sentiment: bool):
    """Run the complete scraping and analysis pipeline."""
    click.echo("Starting polymarket-sentiment pipeline...")
    
    # Run scraper
    if users or terms:
        ctx.invoke(run, users=users, terms=terms, max_tweets=max_tweets)
    
    # Run sentiment analysis
    if analyze_sentiment:
        click.echo("Running sentiment analysis...")
        ctx.invoke(analyze)
    
    click.echo("Pipeline completed successfully!")


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
