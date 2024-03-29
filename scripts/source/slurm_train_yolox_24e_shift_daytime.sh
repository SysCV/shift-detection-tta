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
CONFIG=configs/source/yolox/yolox_x_8xb4-24e_shift_daytime.py
declare -a CFG_OPTIONS=(
     "data.workers_per_gpu=4"
     "data.samples_per_gpu=4"
)
#####
###############

if [ $GPUS -gt 1 ]
then
     CMD=tools/dist_train.sh
else
     CMD=tools/train.sh
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
          --cfg-options ${CFG_OPTIONS[@]})