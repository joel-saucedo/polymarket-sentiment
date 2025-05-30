#!/usr/bin/env bash
set -e

# Simplified ghpush.sh for polymarket-sentiment project
# Usage: ./ghpush.sh "your commit message"
# This script commits changes and pushes directly to main branch

msg=${1:-"Default commit message"}

current_branch=$(git rev-parse --abbrev-ref HEAD)

# Show current changes
echo "Current changes:"
git status --short
echo ""
echo "Current branch: $current_branch"
echo "Commit message: $msg"
echo ""

# Add and commit current changes
echo "Adding and committing changes..."
git add -A
git commit -m "$msg"

# Ensure we are on the main branch
if [ "$current_branch" != "main" ]; then
    echo "Switching to main branch..."
    if git show-ref --verify --quiet refs/heads/main; then
        git checkout main
    else
        echo "Creating main branch..."
        git checkout -b main
    fi
    # If we switched from another branch, merge it.
    if [ "$current_branch" != "main" ]; then # Re-check, as we might have just created main
        echo "Merging $current_branch into main..."
        git merge "$current_branch" --no-ff -m "merge: integrate $current_branch into main"
    fi
else
    echo "Already on main branch."
fi


# Push to origin main (will create remote if doesn't exist)
echo "Pushing to origin main..."
if git remote get-url origin >/dev/null 2>&1; then
    git push origin main
    echo "Successfully pushed to GitHub main branch!"
else
    echo "No remote 'origin' configured."
    echo "To push to GitHub, first add your repository:"
    echo "   git remote add origin https://github.com/yourusername/polymarket-sentiment.git"
    echo "   git push -u origin main"
fi

echo ""
echo "All done! You're now on the main branch with all changes committed and pushed."