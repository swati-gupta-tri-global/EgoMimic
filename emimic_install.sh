#!/bin/bash
# Install emimic environment dependencies using pip

# Set Python version check
python_version=$(python3 --version 2>&1 | grep -o '3\.[0-9]\+')
if [[ "$python_version" != "3.10" ]]; then
    echo "Warning: Python 3.10 recommended, found $python_version"
fi

# Install PyTorch with CUDA 12.1 support
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install \
    "numpy<2.0" \
    pyyaml \
    pexpect \
    matplotlib \
    packaging \
    h5py \
    scipy \
    ipython

# Install ML/AI packages
pip install \
    wandb \
    transformers \
    peft==0.5.0 \
    hydra-core \
    hydra-submitit-launcher \
    einops \
    pytorch-lightning \
    "positional-encodings[pytorch]"

# Install utilities
pip install \
    black \
    gpustat \
    pynvml \
    termcolor \
    pyquaternion \
    rospkg \
    av

# Install OpenCV (specific version)
pip install opencv-python==4.7.0.72

# Install robotics packages
pip install \
    dm-control==1.0.8 \
    mujoco==2.3.1 \
    mujoco-py==2.1.2.14 \
    arm_pytorch_utilities \
    pytorch-kinematics

# Install Git packages
pip install git+https://github.com/simarkareer/submitit
pip install git+https://github.com/ARISE-Initiative/robomimic.git

echo "emimic environment installed successfully!"
echo "Run 'python -c \"import torch; print(torch.cuda.is_available())\"' to test CUDA"