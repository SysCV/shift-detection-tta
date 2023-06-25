# CONFIG=configs/source/yolox/yolox_x_8xb4-24e_shift_clear_daytime.py
CONFIG=configs/source/yolox/amp_yolox_x_8xb4-24e_shift_clear_daytime.py

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/train.py \
python tools/train.py \
     ${CONFIG}