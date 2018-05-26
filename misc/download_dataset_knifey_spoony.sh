DATASET_NAME="knifey_spoony_vanilla"

# Create root dir structure
rm -rf ../datasets/$DATASET_NAME
mkdir ../datasets/$DATASET_NAME
cd ../datasets/$DATASET_NAME

# Download dataset
wget https://github.com/Hvass-Labs/knifey-spoony/raw/master/knifey-spoony.tar.gz
tar -xzvf knifey-spoony.tar.gz 

# Create train and test dir structure
mkdir -p train/forky
mkdir -p train/knifey
mkdir -p train/spoony
mkdir -p test/forky
mkdir -p test/knifey
mkdir -p test/spoony

# Split train and images in different folders
mv forky/*.jpg train/forky/
mv knifey/*.jpg train/knifey/
mv spoony/*.jpg train/spoony/
mv forky/test/*.jpg test/forky/
mv knifey/test/*.jpg test/knifey/
mv spoony/test/*.jpg test/spoony/

# Cleanup
rm -rf forky
rm -rf knifey
rm -rf spoony
