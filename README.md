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
pip install keras
```

## Dataset
Download dataset from https://challenge2018.isic-archive.com/task3/

Update input_dataset_path and input_dataset_labels_file_path variables in the following script and run it
```sh
python scripts/download_dataset_densenet.py -c configs/config_densenet.json
```
Dataset should now be in the following format
```
densenet
├── test
│   ├── AKIEC
│   ├── BCC
│   ├── BKL
│   ├── DF
│   ├── MEL
│   ├── NV
│   └── VASC
└── train
    ├── AKIEC
    ├── BCC
    ├── BKL
    ├── DF
    ├── MEL
    ├── NV
    └── VASC
```

## Generate tensorflow records
```
python data_handler/tfrecords_densenet.py -c configs/config_densenet.json
```

## Train
```sh
./run.sh 1
```

## Evaluate
```sh
./run.sh 2
```

## Predict
```sh
./run.sh 3
```


