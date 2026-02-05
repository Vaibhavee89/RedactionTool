#!/bin/bash
#
# Setup script to install CLI command globally
#
# This script creates a symlink in /usr/local/bin so you can run:
#   redact input_dir/ output_dir/
#
# from anywhere on your system.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PII Redaction Tool - CLI Setup${NC}"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLI_SCRIPT="$SCRIPT_DIR/redact_cli.py"

# Check if CLI script exists
if [ ! -f "$CLI_SCRIPT" ]; then
    echo -e "${RED}Error: CLI script not found at $CLI_SCRIPT${NC}"
    exit 1
fi

# Make CLI script executable
chmod +x "$CLI_SCRIPT"

# Determine install location
if [ -w "/usr/local/bin" ]; then
    INSTALL_DIR="/usr/local/bin"
elif [ -w "$HOME/.local/bin" ]; then
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
else
    echo -e "${YELLOW}Note: Neither /usr/local/bin nor ~/.local/bin is writable${NC}"
    echo "You may need to run with sudo or add to PATH manually"
    INSTALL_DIR="/usr/local/bin"
fi

COMMAND_NAME="redact"
SYMLINK_PATH="$INSTALL_DIR/$COMMAND_NAME"

echo "Installing CLI command..."
echo "  Source: $CLI_SCRIPT"
echo "  Target: $SYMLINK_PATH"
echo ""

# Remove existing symlink if present
if [ -L "$SYMLINK_PATH" ]; then
    echo "Removing existing symlink..."
    rm "$SYMLINK_PATH"
fi

# Create symlink
if [ "$INSTALL_DIR" = "/usr/local/bin" ] && [ ! -w "/usr/local/bin" ]; then
    echo -e "${YELLOW}Requesting sudo privileges to create symlink...${NC}"
    sudo ln -s "$CLI_SCRIPT" "$SYMLINK_PATH"
else
    ln -s "$CLI_SCRIPT" "$SYMLINK_PATH"
fi

# Verify installation
if [ -L "$SYMLINK_PATH" ]; then
    echo -e "${GREEN}✓ CLI command installed successfully!${NC}"
    echo ""
    echo "Usage:"
    echo "  $COMMAND_NAME input_dir/ output_dir/"
    echo "  $COMMAND_NAME --help"
    echo ""
    echo "Example:"
    echo "  $COMMAND_NAME test_data/ output/ --policy policies/india_finance.yaml --mode mask --log audit.json"
    echo ""

    # Check if in PATH
    if command -v "$COMMAND_NAME" &> /dev/null; then
        echo -e "${GREEN}✓ Command is in PATH and ready to use${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: $INSTALL_DIR may not be in your PATH${NC}"
        echo "Add this to your ~/.bashrc or ~/.zshrc:"
        echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
else
    echo -e "${RED}✗ Installation failed${NC}"
    exit 1
fi
