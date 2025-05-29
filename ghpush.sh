#!/usr/bin/env bash
set -e

# Enhanced ghpush.sh for polymarket-sentiment project
# Usage: ./scripts/ghpush.sh "your commit message"

msg=${1:-"feat: complete storage layer implementation with DuckDB/Parquet (Chunk C2)

- Implement TweetModel with Pydantic validation
- Add TweetStorage class with DuckDB integration
- Support Parquet export for data persistence
- Add comprehensive unit tests (25 tests passing)
- Include sentiment data storage and time-series indexing
- Fix SQL compatibility issues and test edge cases

Closes #C2"}
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