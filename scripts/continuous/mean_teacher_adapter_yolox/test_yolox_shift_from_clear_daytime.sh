CONFIG=configs/continuous/mean_teacher_adapter_yolox/yolox_x_8xb4-12e_shift_from_clear_daytime.py
# CONFIG=configs/continuous/mean_teacher_adapter_yolox/amp_yolox_x_8xb4-12e_shift_from_clear_daytime.py
CKPT=checkpoints/yolox_x_8xb4-24e_shift_clear_daytime/20230621_184939/epoch_24.pth

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
python tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT}