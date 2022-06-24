# inspired by https://github.com/ptheywood/cuda-cmake-github-actions
# inspired by https://github.com/tmcdonell/travis-scripts/blob/76755e3dc25e3847501c8730c971d4d2d8d9c1e1/install-cuda-trusty.sh

UBUNTU_VERSION=$(lsb_release -sr)
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

PIN_FILENAME="cuda-ubuntu${UBUNTU_VERSION}.pin"
PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${PIN_FILENAME}"
APT_KEY_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/7fa2af80.pub"
REPO_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/"

CUDA_VER_TMP=$(expr ${CUDA_VER} : '\([0-9]*\.[0-9]*\)')
CUDA_APT=${CUDA_VER_TMP/./-}

# Debug
echo "Ubuntu version ${UBUNTU_VERSION}"
echo "CUDA version ${CUDA_APT}"
echo "PIN_FILENAME ${PIN_FILENAME}"
echo "PIN_URL ${PIN_URL}"
echo "APT_KEY_URL ${APT_KEY_URL}"

# Install
wget ${PIN_URL}
sudo mv ${PIN_FILENAME} /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys ${APT_KEY_URL}
sudo add-apt-repository "deb ${REPO_URL} /"
sudo apt-get update

echo "Installing CUDA"
sudo apt-get install -y cuda-nvcc-${CUDA_APT} cuda-cufft-dev-${CUDA_APT} cuda-cudart-dev-${CUDA_APT} cuda-nvml-dev-${CUDA_APT}


export CUDA_HOME=/usr/local/cuda-${CUDA_VER_TMP}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

