#!/bin/bash
# Code formatting script

set -e

echo "🔧 Running Ruff linter with auto-fix..."
uv run ruff check src/ tests/ --fix

echo "✨ Running Ruff formatter..."
uv run ruff format src/ tests/

echo "🎉 Code formatted successfully!"
