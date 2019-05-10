# Source: https://github.com/tmcdonell/travis-scripts/blob/76755e3dc25e3847501c8730c971d4d2d8d9c1e1/install-cuda-trusty.sh
# Author: Trevor L. McDonell

#!/bin/bash
#
# Install the core CUDA toolkit for a ubuntu-trusty (14.04) system. Requires the
# CUDA environment variable to be set to the required version.
#
# Since this script updates environment variables, to execute correctly you must
# 'source' this script, rather than executing it in a sub-process.
#

travis_retry wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_${CUDA_VER}_amd64.deb
travis_retry sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
travis_retry sudo dpkg -i cuda-repo-ubuntu1404_${CUDA_VER}_amd64.deb
travis_retry sudo apt-get update -qq
export CUDA_VER_TMP=$(expr ${CUDA_VER} : '\([0-9]*\.[0-9]*\)')
export CUDA_APT=${CUDA_VER_TMP/./-}
travis_retry sudo apt-get install -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT}
if [ ${CUDA_INSTALL_EXTRA_LIBS:-1} -ne 0 ]; then
  if [ ${CUDA_VER_TMP%.*} -lt 7 ]; then
    travis_retry sudo apt-get install -y cuda-cufft-dev-${CUDA_APT} cuda-cublas-dev-${CUDA_APT} cuda-cusparse-dev-${CUDA_APT} cuda-curand-dev-${CUDA_APT}
  else
    travis_retry sudo apt-get install -y cuda-cufft-dev-${CUDA_APT} cuda-cublas-dev-${CUDA_APT} cuda-cusparse-dev-${CUDA_APT} cuda-curand-dev-${CUDA_APT}  cuda-cusolver-dev-${CUDA_APT} cuda-nvml-dev-${CUDA_APT}
  fi
fi
travis_retry sudo apt-get clean
export CUDA_HOME=/usr/local/cuda-${CUDA_VER_TMP}
export LD_LIBRARY_PATH=${CUDA_HOME}/nvvm/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

# sudo ldconfig ${CUDA_HOME}/lib64
# sudo ldconfig ${CUDA_HOME}/nvvm/lib64
