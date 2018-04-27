
EXP_NAME='cifar10'
CONFIG_FILE='configs/config_cifar10_small.json'

EXP_FILE='mains/main_densenet.py'


# Train
if [[ $1 -eq 1 ]]; then
	echo "###########"
	echo "#  Train  #"
	echo "###########"
	pkill -9 tensorboard
	rm -vrf output/*
	tensorboard --logdir=../output &
	python $EXP_FILE -m 'train' -c $CONFIG_FILE

# Val
elif [[ $1 -eq 2 ]]; then
	echo "################"
	echo "#  Evaluation  #"
	echo "################"
	# pkill -9 tensorboard
	# rm -rf "output"$EXP_NAME"/summary"
	# tensorboard --logdir=../output &
	python $EXP_FILE -m 'eval' -c $CONFIG_FILE

# Test
elif [[ $1 -eq 3 ]]; then
	echo "################"
	echo "#  Prediction  #"
	echo "################"
	python $EXP_FILE -m 'predict' -c $CONFIG_FILE

else
	echo "Unknown Argument"
	echo "1 for Train, 2 for Evaluation and 3 for Prediction"
fi
