#!/usr/bin/env bash
set -euo pipefail

echo "✨ Running ruff format..."
uv run ruff format .

echo "🔍 Running ruff check..."
uv run ruff check . --fix

echo "✅ Done!"
