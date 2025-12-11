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
wget -O data.tar "https://sooners-my.sharepoint.com/:u:/g/personal/skyan_ou_edu/IQAjMW1JRz7IQYxEttRw83M7Aaxn261mj6vSB-HZ_2LLpQ4?download=1"
tar -xf data.tar

# 3) Third tar (MRMS_precip with 2017–2019)
wget -O data.tar "https://sooners-my.sharepoint.com/:u:/g/personal/skyan_ou_edu/EUNsMkszegBHlFDcdPHVI5oB7p2eGVeRnvr-iK2Y_RTsMw?download=1"
tar -xf data.tar

# Clean up
rm -f data.tar

# Exit data_cali folder
cd ..
