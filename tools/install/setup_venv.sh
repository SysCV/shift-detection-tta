
conda create -n shift-tta python=3.9 -y
conda activate shift-tta

conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch -y

pip install -U openmim
# install mmengine from main branch
python -m pip install git+https://github.com/open-mmlab/mmengine.git@62f9504d701251db763f56658436fd23a586fe25
mim install 'mmcv == 2.0.0rc4'
mim install 'mmdet == 3.0.0rc5'
# install mmclassification from dev-1.x branch at specific commit
python -m pip install git+https://github.com/open-mmlab/mmclassification.git@3ff80f5047fe3f3780a05d387f913dd02999611d
# install mmtracking from dev-1.x branch at specific commit
python -m pip install git+https://github.com/open-mmlab/mmtracking.git@9e4cb98a3cdac749242cd8decb3a172058d4fd6e
python -m pip install git+https://github.com/JonathonLuiten/TrackEval.git
python -m pip install git+https://github.com/scalabel/scalabel.git
python -m pip install --no-input -r requirements.txt

# install shift-detection-tta
pip install --no-input -v -e .
