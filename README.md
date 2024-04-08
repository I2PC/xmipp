[![Build Status](https://github.com/I2PC/xmipp/actions/workflows/build.yml/badge.svg)](https://github.com/I2PC/xmipp/actions/workflows/build.yml)

# Xmipp
<a href="https://github.com/I2PC/xmipp"><img src="https://github.com/I2PC/scipion-em-xmipp/blob/devel/xmipp3/xmipp_logo.png" alt="drawing" width="330" align="left"></a>


  
Xmipp is a suite of image processing programs, primarily aimed at single-particle 3D electron microscopy, designed and managed by the [Biocomputing Unit](http://biocomputingunit.es/) located in Madrid, Spain.

The Xmipp project is divided into four repositories. 
This is the main repository, which contains the majority of the source code for the programs, additional scripts, and tests. Three remaining repositories, [XmippCore](https://github.com/I2PC/xmippCore/), [XmippViz](https://github.com/I2PC/xmippViz/), and [Scipion-em-xmipp](https://github.com/I2PC/scipion-em-xmipp), are automatically downloaded during the installation process.


## Getting started
To have a complete overview about Xmipp please visit the [documentation web](https://i2pc.github.io/docs/). The recommended way for users (not developers) to install and use Xmipp is via the [Scipion](http://scipion.i2pc.es/) framework, where you can use Xmipp with other Cryo-EM-related software. Xmipp  will be installed during Scipion installation (pay attemption on the -noXmipp flag). The Scipion installer should take care of all dependencies for you, however, you can make its life easier if you have your compiler, CUDA and HDF5 library ready and available in the standard paths. Read below for more details about these softwares requirements. Follow the official [installation guide](https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html#installation) of Scipion for more details.

## Installation requirements
### Supported OS
We have tested Xmipp compilation on the following operating systems:
[Ubuntu 18.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-18.04), [Ubuntu 20.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-20.04), [Ubuntu 22.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-22.04), [Centos 7](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-CentOS-7-9.2009). Visit the OS wiki page for more details.

While compilation and execution might be possible on other systems, it might not be straightforward. If you encounter a problem, please refer to known and fixed [issues](https://github.com/I2PC/xmipp/issues?q=is%3Aissue). Let us know if something is not working.

### Hardware requirements
At least 2 processors are required to run Xmipp. In some virtual machine tools only one is assigned, please check that at least two processors are assigned to the virtual machine

### Software dependencies
#### Compiler
Xmipp requires C++17 compatible compiler. We recommend GCC with G++, we have good experience with GCC/G++-8.4 in any case a version >= 8.4 is required. If use GCC/G++-10.3 and CUDA=11 and experience issues, [please change the compiler version](https://github.com/NVIDIA/nccl/issues/494). If use GCC-11 and experience issues, [please visit this](https://github.com/I2PC/xmipp/issues/583). For more details about the compilation proces and installation of gcc, please visit [compiler](https://github.com/I2PC/xmipp/wiki/Compiler)

#### Cmake
Xmipp requires Cmake 3.16 or above. To update it please [visit](https://github.com/I2PC/xmipp/wiki/Cmake-update-and-install)
#### Cuda
Xmipp supports Cuda 8 through 11.7. CUDA is optional but highly recommended. We recommend you to use the newest version available for your operating system, though Cuda 10.2 has the widest support among other Scipion plugins. Pay attemption to the [compiler - CUDA](https://gist.github.com/ax3l/9489132) compatibility.
To install CUDA for your operating system, follow the [official install guide](https://developer.nvidia.com/cuda-toolkit-archive).

#### HDF5
We sometimes see issues regarding the HDF5 dependency. Please visit the [HDF5 Troubleshooting wiki](https://github.com/I2PC/xmipp/wiki/HDF5-Troubleshooting)

#### Full list of dependencies

`sudo apt install -y libfftw3-dev libopenmpi-dev libhdf5-dev python3-numpy python3-dev libtiff5-dev libsqlite3-dev default-jdk git cmake`

Also a compiler will be required (`sudo apt install gcc-10 g++-10`)

## Standalone installation
Standalone installation of Xmipp is recommended for researchers and developers. This installation allows you to use Xmipp without Scipion, however, in the next section it is explained how to link it with Scipion. Xmipp script automatically downloads several dependencies and then creates a configuration file that contains paths and flags used during the compilation. Please refer to the [Xmipp configuration](https://github.com/I2PC/xmipp/wiki/Xmipp-configuration) guide for more info.

Start by cloning the repository and then navigate to the right directory.

`git clone https://github.com/I2PC/xmipp.git xmipp && cd xmipp`

Refer to `./xmipp --help` for additional info on the compilation process and possible customizations.

Next is to compile xmipp. There are to possibilities and in both you will can run Xmipp in Scipion (see [linking step](https://github.com/I2PC/xmipp/edit/agm_refactoring_readme/README.md#linking-xmipp-standalone-to-scipion))
- Compile Xmipp by invoking the compilation script, which will take you through the rest of the process:`./xmipp` that way you will install Xmipp with the dependencies and thier versions that the enviroment you decide, or the default one.
- Compile Xmipp via Scipion enviroment `scipion3 run ./xmipp` that way you will install Xmipp with the dependencies and their versions that Scipion decided. 

As long as you are in a conda enviroment containing a proper installation of Scipion, and variable `XMIPP_LINK_TO_SCIPION` is set to `ON`, the link process will be done automatically by Xmipp's installer.

## Using Xmipp
Xmipp is installed in the build directory located in the same directory where the xmipp script is located. To set all necessary environment variables and paths to all Xmipp programs, you can simply 
`source build/xmipp.bashrc`

## That's all

If you miss some feature or find a bug, please [create an issue](https://github.com/I2PC/xmipp/issues/new) and we will have a look at it, you can also contact us by xmipp@cnb.csic.es. For more details, troubleshootings and information visit the [wikip page.](https://github.com/I2PC/xmipp/wiki)

If you like this project, Watch or Star it! We are also looking for new collaborators, so contact us if you’re interested
