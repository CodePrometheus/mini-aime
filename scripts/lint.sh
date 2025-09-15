#!/bin/bash
# Code quality check script

set -e

echo "🔍 Running Ruff linter..."
uv run ruff check src/ tests/

echo "✨ Running Ruff formatter..."  
uv run ruff format src/ tests/ --check

echo "🧪 Running tests..."
uv run pytest tests/ -v

echo "🎉 All checks passed!"
