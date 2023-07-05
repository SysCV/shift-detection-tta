CONFIG=configs/continuous/no_adap_yolox/yolox_x_8xb4-24e_shift_from_clear_daytime.py
CKPT=https://dl.cv.ethz.ch/shift/challenge2023/test_time_adaptation/checkpoints/yolox_x_8xb4-24e_shift_clear_daytime.pth
WORK_DIR=work_dirs/continuous/no_adap_yolox/val/yolox_x_8xb4-24e_shift_from_clear_daytime

declare -a CFG_OPTIONS=(
     "test_evaluator.0.outfile_prefix=${WORK_DIR}/results"
)

# python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
python tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT} \
     --work-dir ${WORK_DIR} \
     --cfg-options ${CFG_OPTIONS[@]}