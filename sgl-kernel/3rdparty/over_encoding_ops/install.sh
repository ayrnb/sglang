#!/bin/bash
# Install over_encoding_ops independently
# Usage: ./install.sh [--editable]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ "$1" == "--editable" ] || [ "$1" == "-e" ]; then
    echo "Installing over_encoding_ops in editable mode..."
    pip install -e .
else
    echo "Installing over_encoding_ops..."
    pip install .
fi

echo "over_encoding_ops installed successfully!"
