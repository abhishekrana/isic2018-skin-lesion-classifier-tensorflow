CONFIG_FILE='configs/config_densenet.json'
TF_RECORD_FILE='data_handler/tfrecords_densenet.py'
EXP_FILE='mains/main_densenet.py'

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

if [[ $# -eq 0 ]]; then
	echo "ERROR: No argument given."
	echo "0: Generate TF Record"
	echo "1: Train"
	echo "2: Evaluate"
	echo "3: Predict"
	exit 1
fi

# Generate TF Record
if [[ $1 -eq 0 ]]; then
	echo "###############"
	echo "#  TF Record  #"
	echo "###############"
	python $EXP_FILE -m 'tfr' -c $CONFIG_FILE

# Train
elif [[ $1 -eq 1 ]]; then
	echo "###########"
	echo "#  Train  #"
	echo "###########"
	# pkill -9 tensorboard
	# rm -rf output/*
	# tensorboard --logdir=output --port=6004 &
	# export CUDA_VISIBLE_DEVICES=""
	if [[ $2 -eq 1 ]]; then
		python $EXP_FILE -m 'train' -c $CONFIG_FILE -s 'train_ds' -d 1
	else
		python $EXP_FILE -m 'train' -c $CONFIG_FILE -s 'train_ds' -d 0
	fi

# Val
elif [[ $1 -eq 2 ]]; then
	export CUDA_VISIBLE_DEVICES=""
	if [[ $2 -eq 0 ]]; then
		echo "###########################"
		echo "#  Evaluation - train_ds  #"
		echo "###########################"
		python $EXP_FILE -m 'eval' -c $CONFIG_FILE -s 'train_ds'
	else
		echo "#########################"
		echo "#  Evaluation - val_ds  #"
		echo "#########################"
		python $EXP_FILE -m 'eval' -c $CONFIG_FILE -s 'val_ds'
	fi

# Test
elif [[ $1 -eq 3 ]]; then
	export CUDA_VISIBLE_DEVICES=""
	if [[ $2 -eq 1 ]]; then
		echo "#######################"
		echo "#  Prediction Cascade #"
		echo "#######################"
		python $EXP_FILE -m 'predict' -c $CONFIG_FILE -b 1
	else
		echo "################"
		echo "#  Prediction  #"
		echo "################"
		python $EXP_FILE -m 'predict' -c $CONFIG_FILE -b 0
	fi

else
	echo "ERROR: Unknown argument given."
	echo "0: Generate TF Record"
	echo "1: Train"
	echo "2: Evaluate"
	echo "3: Predict"
	exit 1
fi
