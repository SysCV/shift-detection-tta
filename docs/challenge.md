# Workshop on Visual Continual Learning @ ICCV2023

# Challenge on Continual Test-time Adaptation for Object Detection
## Goal

## Rules

## Instructions

### Train a model on the source domain
We train an object detection model on the source domain.

Download our YOLOX checkpoint

### Test the source model on the target domain
We validate the source model on the validation set of the target domain under continuous domain shift.

Continuous sequences


### Continuously adapt a model to the validation target domain
The validation set should be used for validating your method under continuous domain shift and for hyperparameter search.

CustomAdapter and point to config file

### Continuously adapt a model to the test target domain 
Collect your results on the test set and submit to our evaluation [benchmark](). 

Notice that the validation set should be used for validating your method under continuous domain shift and for hyperparameter search. The labels for the test set are indeed kept private to avoid hyperparameter search on the test set.

### Submit your results 

Submit also a report and, optionally, submit your code if you want your adapter included in the repo.

http://34.237.207.206/web/challenges/challenge-page/6/overview