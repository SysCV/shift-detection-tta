
conda create -n shift-tta python=3.9 -y
conda activate shift-tta

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

pip install mmengine

# install the latest mmcv
pip install 'mmcv>=2.0.0rc1' -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# install mmdetection
pip install 'mmdet>=3.0.0rc0'

# install mmtracking
pip install 'mmtrack>=1.0.0rc1'

# install shift-detection-tta
git clone git@github.com:SysCV/shift-detection-tta.git
cd shift-detection-tta
pip install -r requirements/build.txt
pip install -v -e .