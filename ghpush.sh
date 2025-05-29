#!/usr/bin/env bash
set -e

# Simplified ghpush.sh for polymarket-sentiment project
# Usage: ./ghpush.sh "your commit message"
# This script commits changes and pushes directly to main branch

msg=${1:-"feat: complete Chunk C5 - CLI + API endpoints

- Implement comprehensive Click-based CLI with hierarchical commands
- Add FastAPI REST API with 8 endpoints and Pydantic response models  
- Create complete test coverage for both CLI and API (42 tests passing)
- Fix integration issues with AsyncBatchScraper and sentiment processors
- Add dependency injection and error handling throughout
- Update setup.py with all required CLI/API dependencies

All C5 functionality complete and tested. Ready for C6."}

current_branch=$(git rev-parse --abbrev-ref HEAD)

# Show current changes
echo "📋 Current changes:"
git status --short
echo ""
echo "🌿 Current branch: $current_branch"
echo "💬 Commit message: $msg"
echo ""

# Add and commit current changes
echo "📦 Adding and committing changes..."
git add -A
git commit -m "$msg"

# Switch to main branch (create if doesn't exist)
echo "🔄 Switching to main branch..."
if git show-ref --verify --quiet refs/heads/main; then
    git checkout main
else
    echo "📝 Creating main branch..."
    git checkout -b main
fi

# Merge current work if we were on a different branch
if [ "$current_branch" != "main" ]; then
    echo "🔀 Merging $current_branch into main..."
    git merge "$current_branch" --no-ff -m "merge: integrate $current_branch into main"
    
    # Optionally delete the feature branch
    read -p "🗑️  Delete branch '$current_branch'? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git branch -d "$current_branch"
        echo "✅ Deleted branch '$current_branch'"
    fi
fi

# Push to origin main (will create remote if doesn't exist)
echo "🚀 Pushing to origin main..."
if git remote get-url origin >/dev/null 2>&1; then
    git push origin main
    echo "✅ Successfully pushed to GitHub main branch!"
else
    echo "⚠️  No remote 'origin' configured."
    echo "💡 To push to GitHub, first add your repository:"
    echo "   git remote add origin https://github.com/yourusername/polymarket-sentiment.git"
    echo "   git push -u origin main"
fi

echo ""
echo "🎉 All done! You're now on the main branch with all changes committed."