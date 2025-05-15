
conda create -n 3dgsback python=3.10 -y

conda activate 3dgsback
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit=11.8 cuda-nvcc=11.8 -y 
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia


# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.24.4 gsplat==1.4.0 git+https://github.com/ultralytics/CLIP.git  git+https://github.com/krrish94/lseg-minimal.git

# onda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# conda install -c nvidia/label/cuda-12.4.0 cuda-toolkit=12.4 cuda-nvcc=12.4 -y 
pip install tqdm opencv-python scikit-image scikit-learn matplotlib tensorboardX plyfile  open-clip-torch
