#!/bin/bash
# Data Setup Script for Movie Recommender System
# Downloads MovieLens 25M dataset and sets up directory structure

set -e  # Exit on error

echo "=========================================="
echo "Movie Recommender - Data Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if data directory exists
if [ ! -d "data" ]; then
    echo -e "${YELLOW}Creating data directory...${NC}"
    mkdir -p data
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo -e "${YELLOW}Creating models directory...${NC}"
    mkdir -p models
fi

# Check if MovieLens data already exists
if [ -f "data/movies.csv" ] && [ -f "data/ratings.csv" ]; then
    echo -e "${GREEN}✓ MovieLens data already exists${NC}"
    echo ""
    echo "Files found:"
    ls -lh data/*.csv 2>/dev/null || true
    echo ""
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

# Download MovieLens 25M dataset
echo -e "${YELLOW}Downloading MovieLens 25M dataset...${NC}"
echo "This may take a few minutes (size: ~250MB compressed)"
echo ""

DATASET_URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
TEMP_FILE="ml-25m.zip"

# Download with progress
if command -v curl &> /dev/null; then
    curl -L -o "$TEMP_FILE" "$DATASET_URL" --progress-bar
elif command -v wget &> /dev/null; then
    wget -O "$TEMP_FILE" "$DATASET_URL"
else
    echo -e "${RED}Error: Neither curl nor wget found. Please install one of them.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Extracting dataset...${NC}"

# Extract the zip file
if command -v unzip &> /dev/null; then
    unzip -q "$TEMP_FILE"
else
    echo -e "${RED}Error: unzip not found. Please install unzip.${NC}"
    exit 1
fi

# Move files to data directory
echo -e "${YELLOW}Moving files to data directory...${NC}"
mv ml-25m/*.csv data/
mv ml-25m/README.txt data/MOVIELENS_README.txt 2>/dev/null || true

# Clean up
echo -e "${YELLOW}Cleaning up...${NC}"
rm -rf ml-25m
rm "$TEMP_FILE"

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Data setup complete!"
echo "==========================================${NC}"
echo ""
echo "Downloaded files:"
ls -lh data/*.csv
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run the data preparation notebook:"
echo "   cd code && jupyter notebook 01-data-preparation.ipynb"
echo ""
echo "2. Run the model training notebook:"
echo "   cd code && jupyter notebook 02-model_training.ipynb"
echo ""
echo "3. Launch the Streamlit app:"
echo "   cd app && ./run_app.sh"
echo ""

