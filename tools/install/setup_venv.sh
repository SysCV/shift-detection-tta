
conda create -n shift-tta python=3.9 -y
conda activate shift-tta

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch -y

pip install -U openmim
python -m pip install git+ssh://git@github.com/open-mmlab/mmengine.git@main
mim install 'mmcv == 2.0.0rc4'
mim install 'mmdet == 3.0.0rc5'
python -m pip install git+ssh://git@github.com/open-mmlab/mmclassification.git@dev-1.x
python -m pip install git+ssh://git@github.com/open-mmlab/mmtracking.git@dev-1.x
python -m pip install --no-input -r requirements.txt
python -m pip install git+https://github.com/JonathonLuiten/TrackEval.git
python -m pip install git+https://github.com/scalabel/scalabel.git

# install shift-detection-tta
pip install --no-input -v -e .