#!/usr/bin/env bash
# Download and extract three SharePoint tar files sequentially

set -e  # Exit immediately if a command exits with a non-zero status

# Create data_cali folder and enter it
mkdir -p data_cali
cd data_cali

# 1) First tar
wget -O data.tar "https://sooners-my.sharepoint.com/:u:/g/personal/skyan_ou_edu/ERdAMAIviDBBnbK-JBViVjsBSg-Vq4hsiq8_bmhEATVo2w?download=1"
tar -xf data.tar

# 2) Second tar
wget -O data.tar "https://sooners-my.sharepoint.com/:u:/g/personal/skyan_ou_edu/EemEeQkEU5VIuRTidozT4xIBRrSyql21FdFIYj3HE1BvUg?download=1"
tar -xf data.tar

# 3) Third tar
wget -O data.tar "https://sooners-my.sharepoint.com/:u:/g/personal/skyan_ou_edu/EUNsMkszegBHlFDcdPHVI5oB7p2eGVeRnvr-iK2Y_RTsMw?download=1"
tar -xf data.tar

# Clean up
rm -f data.tar

# Exit data_cali folder
cd ..
