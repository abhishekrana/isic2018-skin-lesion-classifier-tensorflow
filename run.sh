
# EXP_NAME='cifar10'
# CONFIG_FILE='configs/config_cifar10_small.json'
# CONFIG_FILE='configs/config_cifar10.json'

CONFIG_FILE='configs/config_densenet.json'
TF_RECORD_FILE='data_handler/tfrecords_densenet.py'
EXP_FILE='mains/main_densenet.py'

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Generate TF Record
if [[ $1 -eq 0 ]]; then
	echo "###############"
	echo "#  TF Record  #"
	echo "###############"
	python $TF_RECORD_FILE -c $CONFIG_FILE

# Train
elif [[ $1 -eq 1 ]]; then
	echo "###########"
	echo "#  Train  #"
	echo "###########"
	pkill -9 tensorboard
	# rm -vrf output/*
	while true; do
		read -p "Save last run output?" response
		case $response in
			[Yy]* ) mv -v output $TIMESTAMP"_output"; break;;
			[Nn]* ) rm -vrf output/*;;
			* ) echo "Please answer y or n.";;
		esac
	done
	tensorboard --logdir=../output &
	python $EXP_FILE -m train -c $CONFIG_FILE

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
