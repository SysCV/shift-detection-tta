#!/bin/bash
TIME=12:00:00  # TIME=(24:00:00)
PARTITION=gpu22  # PARTITION=(gpu16 | gpu20 | gpu22) 
GPUS_TYPE=a40  # GPUS_TYPE=(Quadro_RTX_8000 | a40 | a100)
GPUS=4
CPUS=16
MEM_PER_CPU=22000
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SBATCH_ARGS=${SBATCH_ARGS:-""}
GPUS_PER_NODE=${GPUS}
CPUS_PER_TASK=${CPUS}

###############
##### Your args
CONFIG=configs/continuous/mean_teacher_adapter_yolox/yolox_x_8xb4-24e_shift_from_clear_daytime.py
CKPT=https://dl.cv.ethz.ch/shift/challenge2023/test_time_adaptation/checkpoints/yolox_x_8xb4-24e_shift_clear_daytime.pth
WORK_DIR=work_dirs/continuous/mean_teacher_yolox/test/yolox_x_8xb4-24e_shift_from_clear_ndaytime

declare -a CFG_OPTIONS=(
     "test_evaluator.0.outfile_prefix=${WORK_DIR}/results"
     "test_dataloader.dataset.ann_file=data/shift/continuous/videos/1x/test/front/det_2d_cocoformat.json"
     "test_dataloader.dataset.pipeline.0.backend_args.tar_path=data/shift/continuous/videos/1x/test/front/img_decompressed.tar"
)
#####
###############

if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_test.sh
else
     CMD=tools/test.sh
fi

echo "Launching ${CMD} on ${GPUS} gpus."
echo "Starting job ${JOB_NAME} from ${CONFIG} using --cfg-options ${CFG_OPTIONS[*]}" 

mkdir -p errors/ 
mkdir -p outputs/ 

ID=$(sbatch \
     --parsable \
     -t ${TIME} \
     --job-name=${JOB_NAME} \
     -p ${PARTITION} \
     --gres=gpu:${GPUS_TYPE}:${GPUS_PER_NODE} \
     -e errors/%j.log \
     -o outputs/%j.log \
     --mail-type=BEGIN,END,FAIL \
     ${SBATCH_ARGS} \
     ${CMD} \
          ${CONFIG} \
          ${GPUS} \
          --checkpoint ${CKPT} \
          --cfg-options ${CFG_OPTIONS[@]})

