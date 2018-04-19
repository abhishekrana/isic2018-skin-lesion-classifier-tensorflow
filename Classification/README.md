# Tensorflow_TFRecords_Estimator_Pipeline
A pipeline/template for
- Converting dataset to TFRecords.
- Training and evaluating multi-class image classifier using custom tensorflow estimator.

## Requirements
Tensorflow >= 1.4.0

### Setup Environment
```sh
# Virtual environment (optional)
sudo apt install -y virtualenv

# Tensorflow (optional)
sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
virtualenv --system-site-packages -p python3 tensorflow170_py35_gpu # for Python 3.n with GPU
source tensorflow170_py35_gpu/bin/activate
easy_install -U pip
pip3 install --upgrade tensorflow-gpu # for Python 3.n and GPU

# Dependencies
pip install matplotlib
pip install bunch
pip install pudb
pip install tqdm
```

## Dataset
Download knifey-spoony dataset
```sh
cd scripts
./download_dataset_knifey_spoony.sh
```

## Train and evaluate
```sh
./run.sh
```

## For image classification on new dataset
* Place the new dataset inside datasets folder. Images of each class should be in be in different folder.

Example:
```sh
datasets
  knifey_spoony_vanilla
    train
      forky
      knifey
      spoony
    test
      forky
      knifey
      spoony
```

* Modify configs/config_knifey_spoony.json "labels", "dataset_path_train" and "dataset_path_test" fields.

* Modify models/model_knifey_spoony.py model_fn() as per requirement.

* ./run.sh

## Acknowledgement
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb

https://github.com/MrGemy95/Tensorflow-Project-Template
