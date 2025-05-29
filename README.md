# Polymarket Sentiment Analysis

A real-time sentiment analysis pipeline for generating trading signals on Polymarket, built on top of Nitter scraping.

## 30-Second Executive View

| Goal | Deliverable | Done-When |
|------|-------------|-----------|
| Spin-up self-contained Nitter + ntscraper stack | `docker-compose up` returns healthy containers and `curl http://scraper:8000/health` → OK | [ ] |
| Harden & extend ntscraper | forked repo → new branch → added modules → tests pass | [ ] |
| Produce timestamped tweet store + sentiment index | API `data/tweets.parquet` auto-updating & `/sentiment/latest` returns JSON | [ ] |
| CI/CD & push automation | GH Actions green; `./scripts/ghpush.sh` pushes tagged releases | [x] |

## Features

This system:
- **Scrapes tweets** from Nitter instances (bypassing Twitter API limits)
- **Analyzes sentiment** using VADER and keyword-based scoring  
- **Stores data** in DuckDB/Parquet for fast querying
- **Provides REST API** for real-time sentiment signals
- **Generates trading signals** for Polymarket integration

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone https://github.com/joel-saucedo/ntscraper.git polymarket-sentiment
   cd polymarket-sentiment
   pip install -e .[dev]
   ```

2. **Configure targets:**
   Edit `config/accounts.yml` to specify Twitter accounts and search terms to monitor.

3. **Run the stack:**
   ```bash
   docker-compose up -d
   ```

4. **Check health:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/sentiment/latest
   ```

## Configuration

All configuration is in `config/`:
- `accounts.yml` - Twitter accounts and search terms to monitor
- `scraper.yml` - Rate limits, batch sizes, intervals
- `logging.yml` - Log rotation and levels

Environment variables override config files (prefix: `SCRAPER_`).

## Development

This project follows chunk-wise development (max 300 LOC per PR):

| Chunk | Status | Description |
|-------|--------|-------------|
| C0 | [x] | Repo scaffold + configs + CI skeleton |
| C1 | [ ] | HTTP patches + retries |
| C2 | [ ] | Storage (DuckDB/Parquet) |
| C3 | [ ] | Async batch scraper |
| C4 | [ ] | Sentiment analysis |
| C5 | [ ] | CLI + API endpoints |
| C6 | [ ] | Rate-limit scheduler |
| C7 | [ ] | Metrics + logging |
| C8 | [ ] | Documentation |
| C9 | [ ] | Polymarket hooks |

**Push changes:**
```bash
./scripts/ghpush.sh "feat: your change description"
```

## Architecture

```
Nitter → Scraper → DuckDB/Parquet → Sentiment API → Polymarket
         (async)   (time-series)     (REST)        (signals)
```

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires Docker)
docker-compose up -d
pytest tests/int/ -v
docker-compose down
```

---

## Installation

```
pip install ntscraper
```

## How to use

First, initialize the library:

```python
from ntscraper import Nitter

scraper = Nitter(log_level=1, skip_instance_check=False)
```
The valid logging levels are:
- None = no logs
- 0 = only warning and error logs
- 1 = previous + informational logs (default)

The `skip_instance_check` parameter is used to skip the check of the Nitter instances altogether during the execution of the script. If you use your own instance or trust the instance you are relying on, then you can skip set it to 'True', otherwise it's better to leave it to false.

Then, choose the proper function for what you want to do from the following.

### Scrape tweets

```python
github_hash_tweets = scraper.get_tweets("github", mode='hashtag')

bezos_tweets = scraper.get_tweets("JeffBezos", mode='user')
```

Parameters:
- term: search term
- mode: modality to scrape the tweets. Default is 'term' which will look for tweets containing the search term. Other modes are 'hashtag' to search for a hashtag and 'user' to scrape tweets from a user profile
- number: number of tweets to scrape. Default is -1 (no limit).
- since: date to start scraping from, formatted as YYYY-MM-DD. Default is None
- until: date to stop scraping at, formatted as YYYY-MM-DD. Default is None
- near: location to search tweets from. Default is None (anywhere)
- language: language of the tweets to search. Default is None (any language). The language must be specified as a 2-letter ISO 639-1 code (e.g. 'en' for English, 'es' for Spanish, 'fr' for French ...)
- to: user to which the tweets are directed. Default is None (any user). For example, if you want to search for tweets directed to @github, you would set this parameter to 'github'
- replies: whether to include replies in the search. If 'filters' or 'exclude' are set, this is overridden. Default is False
- filters: list of filters to apply to the search. Default is None. Valid filters are: 'nativeretweets', 'media', 'videos', 'news', 'verified', 'native_video', 'replies', 'links', 'images', 'safe', 'quote', 'pro_video'
- exclude: list of filters to exclude from the search. Default is None. Valid filters are the same as above
- max_retries: max retries to scrape a page. Default is 5
- instance: Nitter instance to use. Default is None and will be chosen at random

Returns a dictionary with tweets and threads for the term.

#### Multiprocessing

You can also scrape multiple terms at once using multiprocessing:

```python
terms = ["github", "bezos", "musk"]

results = scraper.get_tweets(terms, mode='term')
```

Each term will be scraped in a different process. The result will be a list of dictionaries, one for each term.

The multiprocessing code needs to run in a `if __name__ == "__main__"` block to avoid errors. With multiprocessing, only full logging is supported. Also, the number of processes is limited to the number of available cores on your machine.

NOTE: using multiprocessing on public instances is highly discouraged since it puts too much load on the servers and could potentially also get you rate limited. Please only use it on your local instance.

### Get single tweet

```python
tweet = scraper.get_tweet_by_id("x", "1826317783430303888")
```

Parameters:
- username: username of the tweet's author
- tweet_id: ID of the tweet
- instane: Nitter instance to use. Default is None
- max_retries: max retries to scrape a page. Default is 5

Returns a dictionary with the tweet's content.

### Get profile information

```python
bezos_information = scraper.get_profile_info("JeffBezos")
```

Parameters:
- username: username of the page to scrape
- max_retries: max retries to scrape a page. Default is 5
- instance: Nitter instance to use. Default is None
- mode: mode of fetching profile info. 'simple' for basic info, 'detail' for detailed info including following and followers lists. Default is 'simple'

Returns a dictionary of the profile's information.

#### Multiprocessing

As for the term scraping, you can also get info from multiple profiles at once using multiprocessing:

```python
usernames = ["x", "github"]

results = scraper.get_profile_info(usernames)
```

Each user will be scraped in a different process. The result will be a list of dictionaries, one for each user.

The multiprocessing code needs to run in a `if __name__ == "__main__"` block to avoid errors. With multiprocessing, only full logging is supported. Also, the number of processes is limited to the number of available cores on your machine.

NOTE: using multiprocessing on public instances is highly discouraged since it puts too much load on the servers and could potentially also get you rate limited. Please only use it on your local instance.

### Get random Nitter instance

```python
random_instance = scraper.get_random_instance()
```

Returns a random Nitter instance.

## Note

Due to recent changes on Twitter's side, some Nitter instances may not work properly even if they are marked as "working" on Nitter's wiki. If you have trouble scraping with a certain instance, try changing it and check if the problem persists.

## To do list

- [ ] Add scraping of individual posts with comments