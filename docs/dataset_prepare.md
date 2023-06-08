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
│   │   │   │   ├── train
│   │   │   │   │   ├── front
│   │   │   │   │   │   ├── img.zip
│   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   ├── val
│   │   │   │   │   │   ├── img.zip
│   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   ├── continuous
│   │   │   ├── videos
│   │   │   │   ├── 1x
│   │   │   │   │   ├── val
│   │   │   │   │   │   ├── front
│   │   │   │   │   │   │   ├── img.tar
│   │   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   │   ├── test
│   │   │   │   │   │   │   ├── img.tar
│   │   │   │   │   │   │   ├── det_2d.json (the official annotation files)
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
│   │   │   │   ├── train
│   │   │   │   │   ├── front
│   │   │   │   │   │   ├── img.zip
│   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   ├── val
│   │   │   │   │   ├── front
│   │   │   │   │   │   ├── img.zip
│   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   ├── continuous
│   │   │   ├── videos
│   │   │   │   ├── 1x
│   │   │   │   │   ├── val
│   │   │   │   │   │   ├── front
│   │   │   │   │   │   │   ├── img.tar
│   │   │   │   │   │   │   ├── img_decompressed.tar
│   │   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   │   ├── test
│   │   │   │   │   │   ├── front
│   │   │   │   │   │   │   ├── img.tar
│   │   │   │   │   │   │   ├── img_decompressed.tar
│   │   │   │   │   │   │   ├── det_2d.json (the official annotation files)
```

### 2.2 Convert Annotations

We use [CocoVID](https://github.com/open-mmlab/mmtracking/blob/master/mmtrack/datasets/parsers/coco_video_parser.py) to maintain all datasets in this codebase.

In this case, you need to convert the official annotations to this style. We provide scripts and the usages are as following:

```shell
# SHIFT discrete (images, detection-like)
python -m scalabel.label.to_coco -m det -i $DATADIR/shift/discrete/images/$SET_NAME/front/det_2d.json -o $DATADIR/shift/discrete/images/$SET_NAME/front/det_2d_cocoformat.json

# SHIFT continuous (videos, tracking-like)
python -m scalabel.label.to_coco -m box_track -i $DATADIR/shift/continuous/videos/1x/$SET_NAME/front/det_2d.json -o $DATADIR/shift/continuous/videos/1x/$SET_NAME/front/det_2d_cocoformat.json
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
│   │   │   │   ├── train
│   │   │   │   │   ├── front
│   │   │   │   │   │   ├── img.zip
│   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   │   │   ├── det_2d_cocoformat.json (the converted annotation file)
│   │   │   │   ├── val
│   │   │   │   │   ├── front
│   │   │   │   │   │   ├── img.zip
│   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   │   │   ├── det_2d_cocoformat.json (the converted annotation file)
│   │   ├── continuous
│   │   │   ├── videos
│   │   │   │   ├── 1x
│   │   │   │   │   ├── val
│   │   │   │   │   │   ├── front
│   │   │   │   │   │   │   ├── img.tar
│   │   │   │   │   │   │   ├── img_decompressed.tar
│   │   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   │   │   │   ├── det_2d_cocoformat.json (the converted annotation file)
│   │   │   │   │   ├── test
│   │   │   │   │   │   ├── front
│   │   │   │   │   │   │   ├── img.tar
│   │   │   │   │   │   │   ├── img_decompressed.tar
│   │   │   │   │   │   │   ├── det_2d.json (the official annotation files)
│   │   │   │   │   │   │   ├── det_2d_cocoformat.json (the converted annotation file)
```

### 2.3 Dataset Loading
Some high-performance clusters do not support folders with a large number of files. For this reason, we implemented a [ZipBackend](shift_tta/fileio/backends/zip_backend.py) and a [TarBackend](shift_tta/fileio/backends/tar_backend.py) for loading data directly from `.zip` and `.tar` files.

For usage, refer to the [`shift.py`](configs/_base_/datasets/shift.py) config file.

# TODO: we might have to split the dataset file into two, depending on how we want to handle testing on the discrete val set and adapting to the continuous val set.