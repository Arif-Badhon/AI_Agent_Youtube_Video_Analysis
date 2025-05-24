#!/bin/bash
echo "Running flake8 linter..."
flake8 src/

echo "Auto-formatting with black..."
black src/

echo "Checking for outdated packages..."
pip list --outdated

echo "Running tests..."
pytest

echo "Done!"