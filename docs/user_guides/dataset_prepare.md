## SHIFT Dataset Preparation

This page provides the instructions for the [SHIFT](https://www.vis.xyz/shift/) dataset preparation.

### 1. Downloading the Dataset

Please download the SHIFT dataset from the [official website](https://www.vis.xyz/shift/download/) to your $DATADIR. It is recommended to symlink the root of the datasets to `$SHIFT_DETECTION_TTA/data`. This will avoid storing large files in your project directory, a requirement of several high-performance computing systems.

Examples of other directories that we recommend to symlink are `checkpoints/`, `data/`, `work_dir/`.

Symlink your data directory to the `$SHIFT_DETECTION_TTA` base directory using:

```shell
ln -s $DATADIR/ $SHIFT_DETECTION_TTA/
```

Then, use the official [download.py](https://github.com/SysCV/shift-dev/blob/main/download.py) script provided with the SHIFT devkit to download the dataset. 

```shell
mkdir -p $DATADIR/shift

# Download the discrete shift set for training source models
python tools/shift/download.py \
    --view "[front]" --group "[img, det_2d]" \
    --split "[train, val]" --framerate "[images]" \
    --shift "discrete" \
    $DATADIR/shift

# Download the continuous shift set for test-time adaptation
python tools/shift/download.py \
    --view "[front]" --group "[img, det_2d]" \
    --split "[val, test]" --framerate "[videos]" \
    --shift "continuous/1x" \
    $DATADIR/shift
```

#### 1.1 Data Structure

We here report the recommended data structure. If your folder structure is different from the following, you may need to change the corresponding paths in the config files.

```
shift-detection-tta
├── shift_tta
├── tools
├── configs
├── data
│   ├── shift
│   │   ├── discrete
│   │   │   ├── images
│   │   ├── continuous
│   │   │   ├── DET
│   │   ├── annotations
```


### 2. Process the Dataset

### 2.1 Decompress the Dataset
To ensure reproducible decompression of videos, we recommend using the [Docker image](https://github.com/SysCV/shift-dev/blob/main/Dockerfile) from the [official SHIFT devkit](https://github.com/SysCV/shift-dev). You could refer to the Docker engine's installation doc.

```shell
# clone the SHIFT devkit
git clone git@github.com:SysCV/shift-dev.git
cd shift-dev

# build and install our Docker image
docker build -t shift_dataset_decompress .

# run the container (the mode is set to "hdf5")
docker run -v <path/to/data>:/data -e MODE=hdf5 shift_dataset_decompress
Here, <path/to/data> denotes the root path under which all tar files will be processed recursively. The mode and number of jobs can be configured through environment variables MODE and JOBS.
```


### 2.2 Convert Annotations

We use [CocoVID](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase.
In this case, you need to convert the official annotations to this style. We provide scripts and the usages are as following:

```shell
# SHIFT discrete (images, detection-like)
mkdir -p $DATADIR/shift/discrete/images/$SET_NAME/front/
python -m scalabel.label.to_coco -m det -i $DATADIR/shift/labels/det_20/det_$SET_NAME.json -o $DATADIR/shift/annotations/det_20/box_det_$SET_NAME_cocofmt.json

# SHIFT continuous (videos, tracking-like)
mkdir -p $DATADIR/shift/continuous/videos/$SET_NAME/front/
python -m scalabel.label.to_coco -m box_track -i $DATADIR/shift/labels/box_track_20/$SET_NAME -o $DATADIR/shift/annotations/box_track_20/box_track_$SET_NAME_cocofmt.json
```

where `$SET_NAME` is one of `[train, val, test]`.

The folder structure will be as following after your run these scripts:

```
shift-detection-tta
├── shift_tta
├── tools
├── configs
├── data
│   ├── shift
│   │   ├── discrete
│   │   │   ├── images
│   │   ├── continuous
│   │   │   ├── DET
│   │   ├── annotations
│   │
│   ├── youtube_vis_2021
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │   │── ......
│   │   │── annotations (the converted annotation file)
```


#### The folder of annotations in youtube_vis_2019/youtube_vis2021

There are 3 JSON files in `data/youtube_vis_2019/annotations` or `data/youtube_vis_2021/annotations`:

`youtube_vis_2019_train.json`/`youtube_vis_2021_train.json`: JSON file containing the annotations information of the training set in youtube_vis_2019/youtube_vis2021 dataset.

`youtube_vis_2019_valid.json`/`youtube_vis_2021_valid.json`: JSON file containing the annotations information of the validation set in youtube_vis_2019/youtube_vis2021 dataset.

`youtube_vis_2019_test.json`/`youtube_vis_2021_test.json`: JSON file containing the annotations information of the testing set in youtube_vis_2019/youtube_vis2021 dataset.