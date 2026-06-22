#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?Usage: release.sh <version>}"

echo "Releasing fuzzer-tool v${VERSION}"

# Update version in pyproject.toml
sed -i "s/^version = \".*\"/version = \"${VERSION}\"/" pyproject.toml

# Update version in __init__.py
sed -i "s/__version__ = \".*\"/__version__ = \"${VERSION}\"/" src/fuzzer_tool/__init__.py

# Run checks
echo "Running linter..."
ruff check src/ tests/

echo "Running type checker..."
mypy src/

echo "Running tests..."
pytest

# Commit and tag
git add pyproject.toml src/fuzzer_tool/__init__.py
git commit -m "Release v${VERSION}"
git tag "v${VERSION}"

echo "Tagged v${VERSION}. Run: git push && git push --tags"
