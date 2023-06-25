CONFIG=configs/source/yolox/yolox_x_8xb4-12e_shift_clear_daytime.py
CKPT=https://dl.cv.ethz.ch/shift/challenge2023/test_time_adaptation/checkpoints/yolox_x_8xb4-24e_shift_clear_daytime.pth

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
python tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT}