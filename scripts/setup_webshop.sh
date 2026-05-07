#!/bin/bash

# Exit on error
set -e

echo "Setting up webshop..."
echo "NOTE: please run scripts/setup_ragen.sh before running this script"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
# TODO: merge this with the main setup script with an option to install webshop
# Install if you want to use webshop
conda install -c pytorch faiss-cpu -y
sudo apt update
sudo apt install default-jdk
conda install -c conda-forge openjdk=21 maven -y

# Install remaining requirements
print_step "Installing additional requirements..."
pip install -r requirements.txt

# webshop installation, model loading
pip install -e external/webshop-minimal/ --no-dependencies
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

print_step "Downloading data..."
python scripts/download_data.py

# Optional: download full data set
print_step "Downloading full data set..."
conda install conda-forge::gdown
mkdir -p external/webshop-minimal/webshop_minimal/data/full
cd external/webshop-minimal/webshop_minimal/data/full
gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB # items_shuffle
gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi # items_ins_v2
cd ../../../../..

echo -e "${GREEN}Installation completed successfully!${NC}"
echo "To activate the environment, run: conda activate ragen"

