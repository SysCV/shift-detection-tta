CONFIG=configs/source/yolox/yolox_x_8xb4-12e_shift_clear_daytime.py
CKPT=work_dirs/yolox_x_8xb4-12e_shift_clear_daytime/epoch_1.pth

# python tools/test.py \
python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT}