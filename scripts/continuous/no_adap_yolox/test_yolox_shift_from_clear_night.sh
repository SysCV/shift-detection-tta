CONFIG=configs/continuous/no_adap_yolox/yolox_x_8xb4-12e_shift_from_clear_night.py
CKPT=checkpoints/yolox_x_8xb4-24e_shift_clear_daytime/20230621_184939/epoch_24.pth
WORK_DIR=work_dirs/continuous/no_adap_yolox/yolox_x_8xb4-12e_shift_from_clear_night

declare -a CFG_OPTIONS=(
     "test_evaluator.0.outfile_prefix=${WORK_DIR}/results"
     "test_dataloader.dataset.ann_file=data/shift/continuous/videos/1x/test/front/det_2d_cocoformat_tmp.json"
     "test_dataloader.dataset.pipeline.0.backend_args.tar_path=data/shift/continuous/videos/1x/test/front/img_decompressed.tar"

)

# python tools/test.py \
python -m debugpy --listen $HOSTNAME:5678 --wait-for-client tools/test.py \
     ${CONFIG} \
     --checkpoint ${CKPT} \
     --work-dir ${WORK_DIR} \
     --cfg-options ${CFG_OPTIONS[@]}