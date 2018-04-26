## Cleanup
pkill -9 tensorboard
sleep 1

rm -vrf output/*

tensorboard --logdir=output &
sleep 1


## Run training
# python mains/main_knifey_spoony.py -c configs/config_knifey_spoony.json
# python mains/main_cifar10.py -c configs/config_cifar10.json
python mains/main_densenet.py -c configs/config_cifar10.json
