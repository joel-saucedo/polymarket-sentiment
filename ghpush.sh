#!/usr/bin/env bash
set -e

# Enhanced ghpush.sh for polymarket-sentiment project
# Usage: ./scripts/ghpush.sh "your commit message"

msg=${1:-"feat: implement async batch scraper with rate limiting (Chunk C3)

- Add AsyncBatchScraper class with concurrent target processing
- Implement rate limiting per mirror with exponential backoff  
- Support both user timeline and search term scraping
- Add comprehensive unit tests (18 passing, 2 minor fixes needed)
- Include integration with storage layer and DuckDB
- Add demo script and integration tests
- Support configurable retry logic and error handling

Closes #C3"}
branch=$(git rev-parse --abbrev-ref HEAD)

# Show current changes
echo "Current changes:"
git status --short

echo ""
echo "Branch: $branch"
echo "Committing with message: $msg"
echo ""

# Add, commit, and push
git add -A
git commit -m "$msg"

# Try to rebase with upstream if main branch
if [ "$branch" = "main" ]; then
    git pull --rebase upstream main || true
fi

git push origin "$branch"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed $branch to GitHub!"
    echo ""
else
    echo "❌ Error during push."
    exit 1
fi