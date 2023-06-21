CONFIG_FILE=configs/source/yolox/yolox_x_8xb4-12e_shift.py

# python tools/train.py \
python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/train.py \
     ${CONFIG_FILE}