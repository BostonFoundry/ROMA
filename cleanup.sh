#!/bin/bash

echo "🔪 Killing processes for user 'mshiffer' matching '.pyenv'..."
ps -u mshiffer -f | grep .pyenv | grep -v grep | awk '{print $2}' | xargs -r kill

echo "🔪 Killing processes for user 'mshiffer' matching 'roma'..."
ps -u mshiffer -f | grep roma | grep -v grep | awk '{print $2}' | xargs -r kill

sleep 1  # 💤 Wait for 1 second

echo "📋 Listing remaining processes for user 'mshiffer'..."
ps -u mshiffer

echo "⬇️ Performing git pull..."
git pull

echo "✅ Script complete."

