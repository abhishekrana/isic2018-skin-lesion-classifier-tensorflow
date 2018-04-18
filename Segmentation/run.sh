# Cleanup
# rm -vrf output/knifey_spoony/*

# Run training
# python mains/main_knifey_spoony.py -c configs/config_knifey_spoony.json
python mains/main_unet.py -c configs/config_unet.json
