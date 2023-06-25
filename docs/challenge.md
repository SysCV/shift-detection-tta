# Workshop on Visual Continual Learning @ ICCV2023

# Challenge on Continual Test-time Adaptation for Object Detection
## Goal
We introduce the [1st Challenge on Continual Test-time Adaptation for Object Detection](https://wvcl.vis.xyz/challenges).

The goal of this challenge is training an object detector on the SHIFT clear-daytime subset (source domain) and adapting it to the set of SHIFT sequences with continuous domain shift starting from clear-daytime conditions.

## Rules
- Using additional data is **not** allowed;
- Any detector architecture can be used;
- The model should be adapted on the fly to each target sequence, and reset to its original state at the end of every sequence.

You can find a reference implementation for an [AdaptiveDetector](shift_tta/models/detectors/adaptive_detector.py) class wrapping any object detector and an adapter, a [BaseAdapter](shift_tta/models/adapters/base_adapter.py) class and a reference implementation of a [mean-teacher adapter](shift_tta/models/adapters/mean_teacher_adapter_yolox.py) based on YOLOX.

## Instructions

### Train a model on the source domain
First, train an object detection model on the source domain. You may choose any object detector architecture.

You can find a reference training script at [scripts/source/train_yolox_shift_clear_daytime.sh](scripts/source/train_yolox_shift_clear_daytime.sh) to train a YOLOX model on the SHIFT clear-daytime discrete set.

We use the discrete set of SHIFT to train the object detector.

You can also download a YOLOX checkpoint pre-trained using the above-mentioned script at [link](https://dl.cv.ethz.ch/shift/challenge2023/test_time_adaptation/checkpoints/yolox_x_8xb4-24e_shift_clear_daytime.pth).

### Test the source model on the target domain
Then, validate the source model on the validation set of the continuous target domain. In particular, we validate on the videos presenting continuous domain shift starting from the clear-daytime conditions. The validation set should be used for validating your method under continuous domain shift and for hyperparameter search.

You can find a reference validation script at [scripts/continuous/no_adap_yolox/val_yolox_shift_from_clear_daytime.sh](scripts/continuous/no_adap_yolox/val_yolox_shift_from_clear_daytime.sh).


### Continuously adapt a model to the validation target domain
You can now validate your test-time adaptation baseline on the validation videos presenting continuous domain shift starting from the clear-daytime conditions. The validation set should be used for validating your method under continuous domain shift and for hyperparameter search.

We implemented a baseline adapter based on a detection consistency loss and a mean-teacher formulation. You can find an implementation of the adapter at [mean_teacher_yolox_adapter](shift_tta/models/adapters/mean_teacher_yolox_adapter.py), and the corresponding config file at [configs/continuous/mean_teacher_adapter_yolox/yolox_x_8xb4-12e_shift_from_clear_daytime.py](configs/continuous/mean_teacher_adapter_yolox/yolox_x_8xb4-12e_shift_from_clear_daytime.py).

You can run the adaptation script on the validation set using [scripts/continuous/mean_teacher_adapter_yolox/val_yolox_shift_from_clear_daytime.sh](scripts/continuous/mean_teacher_adapter_yolox/val_yolox_shift_from_clear_daytime.sh)

### Continuously adapt a model to the test target domain 
Finally, collect your results on the test set and submit to our evaluation [benchmark](https://evalai.vis.xyz/web/challenges/challenge-page/6/overview). 

You can now test your test-time adaptation baseline on the test videos presenting continuous domain shift starting from the clear-daytime conditions.

We implemented a baseline adapter based on a detection consistency loss and a mean-teacher formulation. You can find an implementation of the adapter at [mean_teacher_yolox_adapter](shift_tta/models/adapters/mean_teacher_yolox_adapter.py), and the corresponding config file at [configs/continuous/mean_teacher_adapter_yolox/yolox_x_8xb4-12e_shift_from_clear_daytime.py](configs/continuous/mean_teacher_adapter_yolox/yolox_x_8xb4-12e_shift_from_clear_daytime.py). 

You can run the adaptation script on the validation set using [scripts/continuous/mean_teacher_adapter_yolox/test_yolox_shift_from_clear_daytime.sh](scripts/continuous/mean_teacher_adapter_yolox/test_yolox_shift_from_clear_daytime.sh)

### Submit your results 


Running the above-mentioned scripts with the following `CFG_OPTIONS` stores results in the [Scalabel](https://www.scalabel.ai/) format in `${WORK_DIR}/results`:

```bash
declare -a CFG_OPTIONS=(
     "test_evaluator.0.outfile_prefix=${WORK_DIR}/results"
)
```

Identify the file ending with `.scalabel.json` and submit it to our [evaluation benchmark](https://evalai.vis.xyz/web/challenges/challenge-page/6/overview) to participate in the challenge.

We require participants to submit a short report providing details on their solution. Optionally, participant may submit their code or open a pull request after the challenge deadline if they want their adapter included in this repository.
