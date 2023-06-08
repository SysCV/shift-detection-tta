
conda create -n shift-tta python=3.9 -y
conda activate shift-tta

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch -y

pip install --no-input mmengine

# install the latest mmcv
pip install --no-input 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# install mmdetection
pip install --no-input 'mmdet>=3.0.0rc0'

# install mmtracking
pip install --no-input 'mmtrack>=1.0.0rc1' 

# install shift-detection-tta
pip install --no-input -r requirements/build.txt
pip install --no-input -v -e .