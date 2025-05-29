"""
REST API for polymarket-sentiment pipeline.

This module provides FastAPI endpoints for:
- Health checks
- Sentiment data access
- Real-time trading signals
- System status and metrics
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from polymkt_sent.core.storage import TweetStorage, StorageConfig
from polymkt_sent.core.sentiment import SentimentProcessor, SentimentConfig, create_sentiment_processor


logger = logging.getLogger(__name__)


# Response Models
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component statuses")


class SentimentResponse(BaseModel):
    """Sentiment analysis response model."""
    tweet_id: str = Field(..., description="Tweet identifier")
    content: str = Field(..., description="Tweet content")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score")
    sentiment_label: str = Field(..., description="Sentiment label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: datetime = Field(..., description="Tweet timestamp")
    username: str = Field(..., description="Tweet author")


class SentimentSummaryResponse(BaseModel):
    """Sentiment summary response model."""
    total_tweets: int = Field(..., description="Total tweets analyzed")
    positive_tweets: int = Field(..., description="Positive sentiment count")
    negative_tweets: int = Field(..., description="Negative sentiment count")
    neutral_tweets: int = Field(..., description="Neutral sentiment count")
    average_sentiment: float = Field(..., description="Average sentiment score")
    last_updated: datetime = Field(..., description="Last update timestamp")


class TradingSignalResponse(BaseModel):
    """Trading signal response model."""
    signal: str = Field(..., description="Trading signal (BUY/SELL/HOLD)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Signal confidence")
    reasoning: str = Field(..., description="Reasoning for the signal")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment")
    tweet_count: int = Field(..., description="Number of tweets analyzed")
    timestamp: datetime = Field(..., description="Signal generation time")


# API Configuration
class APIConfig:
    """API configuration settings."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.storage_config = StorageConfig(data_dir=data_dir)
        self.sentiment_config = SentimentConfig()
        
    @property
    def storage(self) -> TweetStorage:
        """Get storage instance."""
        return TweetStorage(self.storage_config)
    
    @property
    def sentiment_processor(self) -> SentimentProcessor:
        """Get sentiment processor instance."""
        return SentimentProcessor(self.storage, self.sentiment_config)


# Global config instance
api_config = APIConfig()


# FastAPI app
app = FastAPI(
    title="Polymarket Sentiment API",
    description="Real-time sentiment analysis API for Polymarket trading signals",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Dependency injection
def get_storage() -> TweetStorage:
    """Dependency to get storage instance."""
    return TweetStorage(api_config.storage_config)


def get_sentiment_processor() -> SentimentProcessor:
    """Dependency to get sentiment processor instance."""
    storage = TweetStorage(api_config.storage_config)
    return SentimentProcessor(storage, api_config.sentiment_config)


# Health endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check(storage: TweetStorage = Depends(get_storage), processor: SentimentProcessor = Depends(get_sentiment_processor)):
    """Health check endpoint."""
    try:
        # Check storage connection
        storage_status = "healthy"
        try:
            # Test storage with a simple query
            storage.get_tweet_count()
        except Exception as e:
            storage_status = f"unhealthy: {str(e)}"
            logger.warning(f"Storage health check failed: {e}")
        
        # Check data directory
        data_status = "healthy" if Path(api_config.data_dir).exists() else "missing"
        
        # Check sentiment processor
        sentiment_status = "healthy"
        try:
            # Quick validation
            if not processor.analyzer:
                sentiment_status = "no analyzer"
        except Exception as e:
            sentiment_status = f"unhealthy: {str(e)}"
            logger.warning(f"Sentiment processor health check failed: {e}")
        
        overall_status = "healthy" if all(
            "unhealthy" not in status and status not in ["missing", "no analyzer"] 
            for status in [storage_status, data_status, sentiment_status]
        ) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            components={
                "storage": storage_status,
                "data_directory": data_status,
                "sentiment_processor": sentiment_status
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Sentiment endpoints
@app.get("/sentiment/latest", response_model=List[SentimentResponse])
async def get_latest_sentiment(
    limit: int = Query(10, ge=1, le=100, description="Number of tweets to return"),
    storage: TweetStorage = Depends(get_storage)
):
    """Get latest tweets with sentiment analysis."""
    try:
        tweets = storage.get_tweets(with_sentiment=True, limit=limit)
        
        responses = []
        for tweet in tweets:
            if all(key in tweet for key in ['tweet_id', 'content', 'sentiment_score', 'sentiment_label']):
                responses.append(SentimentResponse(
                    tweet_id=tweet['tweet_id'],
                    content=tweet['content'],
                    sentiment_score=tweet['sentiment_score'],
                    sentiment_label=tweet['sentiment_label'],
                    confidence=tweet.get('sentiment_confidence', 0.0),
                    timestamp=tweet['timestamp'],
                    username=tweet['username']
                ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to get latest sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sentiment data: {str(e)}")


@app.get("/sentiment/summary", response_model=SentimentSummaryResponse)
async def get_sentiment_summary(
    processor: SentimentProcessor = Depends(get_sentiment_processor)
):
    """Get sentiment analysis summary."""
    try:
        summary = processor.get_sentiment_summary()
        
        if not summary or (isinstance(summary, dict) and summary.get("error")):
            error_msg = summary.get("error", "No data available") if isinstance(summary, dict) else "No data available"
            raise HTTPException(status_code=500, detail=error_msg)
        
        return SentimentSummaryResponse(
            total_tweets=summary.get('total_tweets', 0),
            positive_tweets=summary.get('positive_tweets', 0),
            negative_tweets=summary.get('negative_tweets', 0),
            neutral_tweets=summary.get('neutral_tweets', 0),
            average_sentiment=summary.get('avg_sentiment', 0.0),
            last_updated=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Failed to get sentiment summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment summary: {str(e)}")


@app.get("/signals/trading", response_model=TradingSignalResponse)
async def get_trading_signal(
    lookback_hours: int = Query(24, ge=1, le=168, description="Hours to look back for signal"),
    processor: SentimentProcessor = Depends(get_sentiment_processor)
):
    """Generate trading signal based on recent sentiment."""
    try:
        # Get sentiment summary
        summary = processor.get_sentiment_summary()
        
        if not summary or (isinstance(summary, dict) and summary.get("error")):
            error_msg = summary.get("error", "No data available") if isinstance(summary, dict) else "No data available"
            raise HTTPException(status_code=500, detail=error_msg)
        
        total_tweets = summary.get('total_tweets', 0)
        if total_tweets == 0:
            return TradingSignalResponse(
                signal="HOLD",
                confidence=0.0,
                reasoning="Insufficient data: no tweets available",
                sentiment_score=0.0,
                tweet_count=0,
                timestamp=datetime.now(timezone.utc)
            )
        
        avg_sentiment = summary.get('avg_sentiment', 0.0)
        positive_ratio = summary.get('positive_tweets', 0) / total_tweets
        negative_ratio = summary.get('negative_tweets', 0) / total_tweets
         # Simple trading signal logic
        if avg_sentiment > 0.3 and positive_ratio > 0.6:
            signal = "BUY"
            confidence = min(0.9, avg_sentiment + positive_ratio * 0.5)
            reasoning = f"Strong positive sentiment ({avg_sentiment:.2f}) with {positive_ratio:.1%} positive tweets"
        elif avg_sentiment < -0.3 and negative_ratio > 0.6:
            signal = "SELL"
            confidence = min(0.9, abs(avg_sentiment) + negative_ratio * 0.5)
            reasoning = f"Strong negative sentiment ({avg_sentiment:.2f}) with {negative_ratio:.1%} negative tweets"
        else:
            signal = "HOLD"
            confidence = 0.5
            reasoning = f"Mixed sentiment ({avg_sentiment:.2f}) suggests holding position"

        return TradingSignalResponse(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            sentiment_score=avg_sentiment,
            tweet_count=total_tweets,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Failed to generate trading signal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate trading signal: {str(e)}")


# System endpoints
@app.get("/stats")
async def get_system_stats(
    storage: TweetStorage = Depends(get_storage)
) -> Dict[str, Any]:
    """Get system statistics."""
    try:
        stats = {
            "total_tweets": storage.get_tweet_count(),
            "uptime": "unknown",  # TODO: Track uptime
            "last_scrape": "unknown",  # TODO: Get from scraper status
            "api_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system statistics: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Server runner
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "polymkt_sent.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
