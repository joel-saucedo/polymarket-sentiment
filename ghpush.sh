#!/usr/bin/env bash
set -e

# Enhanced ghpush.sh for polymarket-sentiment project
# Usage: ./scripts/ghpush.sh "your commit message"

msg=${1:-"feat: complete sentiment analysis framework with VADER and keyword analyzers

- Implement modular sentiment analysis with SentimentResult and SentimentConfig data classes
- Add VADERAnalyzer and KeywordAnalyzer with crypto-specific keywords
- Create EnsembleSentimentAnalyzer with weighted scoring and batch processing
- Include comprehensive unit tests with 37 passing tests and full coverage
- Add demo script showcasing individual analyzers, ensemble analysis, and async processing
- Fix VADER import path and TweetStorage integration issues

Closes #C4"}
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