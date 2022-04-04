[![Build Status](https://github.com/I2PC/xmipp/actions/workflows/main.yml/badge.svg)](https://github.com/I2PC/xmipp/actions/workflows/main.yml)

# Xmipp

Welcome to Xmipp. 
Xmipp is a suite of image processing programs, primarily aimed at single-particle 3D electron microscopy, designed and managed by the [Biocomputing Unit](http://biocomputingunit.es/) located in Madrid, Spain.

The Xmipp project is divided into four repositories. 
This is the main repository, which contains the majority of the source code for the programs, additional scripts, and tests. Three remaining repositories, [XmippCore](https://github.com/I2PC/xmippCore/), [XmippViz](https://github.com/I2PC/xmippViz/), and [Scipion-em-xmipp](https://github.com/I2PC/scipion-em-xmipp), are automatically downloaded during the compilation/installation process.

If you like this project, Watch or Star it!

We are also looking for new collaborators, so contact us if you’re interested!

# Getting started
The recommended way for users (not developers) to install and use Xmipp is via the [Scipion](http://scipion.i2pc.es/) framework, where you can use Xmipp with other Cryo-EM-related software.

Xmipp consists of multiple standalone programs written primarily in C++. For that reason, the Xmipp suite has to be compiled before its use. The compilation process is driven by the xmipp script located in this repository. 

Xmipp script automatically downloads several dependencies and then creates a configuration file that contains paths and flags used during the compilation. Please refer to the [Xmipp configuration](https://github.com/I2PC/xmipp/wiki/Xmipp-configuration-(version-20.07)) guide for more info.

Have a look at the Supported OS and dependencies below, before you install Xmipp as a Scipion package or as a standalone installation. You can also Link your Standalone version to Scipion later.

## Supported OS
We have tested Xmipp compilation on the following operating systems:
- [Installation guide for Ubuntu 16.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-16.04)
- [Installation guide for Ubuntu 18.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-18.04)
- [Installation guide for Ubuntu 20.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-20.04)
- [Installation guide for Centos 7](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-CentOS-7-9.2009)

While compilation and execution might be possible on other systems, it might not be straightforward. If you encounter a problem, please refer to known and fixed [issues](https://github.com/I2PC/xmipp/issues?q=is%3Aissue). Let us know if something is not working!

## Hardware requirements
At least 2 processors are required to run Xmipp. In some virtual machine tools only one is assigned, please check that at least two processors are assigned to the virtual machine

## Additional dependencies
### Compiler
Xmipp requires C++17 compatible compiler. We recommend either GCC or CLANG, in the newest version possible. We have good experience with GCC-8 and bad experience with GCC-7.

We strongly recommend you to have this compiler linked to `gcc` and `g++`. Otherwise it might not be properly picked up by wrappers, such as MPI's wrapper.
We have good experince with using `alternatives`:

```
sudo apt install gcc-8 g++-8
sudo update-alternatives --remove-all gcc
sudo update-alternatives --remove-all g++
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 50
```

### Cuda
Xmipp supports Cuda 8 through 11. CUDA is optional but highly recommended. We recommend you to use the newest version available for your operating system, though Cuda 10.2 has the widest support among other Scipion plugins.
To install CUDA for your operating system, follow the [official install guide](https://developer.nvidia.com/cuda-toolkit-archive).

### OpenCV
OpenCV is used for several programs, however, it is not required.
If you installed OpenCV via apt (`sudo apt install libopencv-dev`), it should be automatically picked up by the Xmipp script

### HDF5
We sometimes see issues regarding the HDF5 dependency.
We strongy recommend you to install it via your default package manager:
`sudo apt-get install libhdf5-dev` 
If you install it using other package management system (such as Conda), it might lead to compile/link time issues caused by incompatible version being fetched.

### Full list of dependencies
`sudo apt install -y libfftw3-dev libopenmpi-dev libhdf5-dev python3-numpy python3-dev libtiff5-dev libsqlite3-dev default-jdk git cmake gcc-8 g++-8`

`pip install scons`

# Installing Xmipp as a Scipion plugin
This is a recommended way for end users.
Xmipp can (and will) be installed during Scipion installation. The Scipion installer should take care of all dependencies for you, however, you can make its life easier if you have your compiler, CUDA and HDF5 library ready and available in the standard paths.

Follow the official [installation guide](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html#) of Scipion for more details.


# Standalone installation
Standalone installation of Xmipp is recommended for researchers and developers. This installation allows you to use Xmipp without Scipion. However, in the next section it is explained how to link it with Scipion.
Start by cloning the repository and then navigate to the right directory.

`git clone https://github.com/I2PC/xmipp.git xmipp-bundle && cd xmipp-bundle`

You might want to change the branch at this moment, however, the default branch is recommended, as it contains the latest and greatest.

Next is to compile xmipp. There are to possibilities. 1) Compile Xmipp by invoking the compilation script, which will take you through the rest of the process:`./xmipp` or alternatively 2) Compile Xmipp via Scipion `scipion3 run xmipp`. To do that we need Scipion installed ([see Scipion installation web page](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html#))

It is important to highlight that this step only compiles Xmipp, but it does not link to Scipion. The linking to Scipion is explained in the next section.

Please refer to `./xmipp --help` for additional info on the compilation process and possible customizations.


# Linking standalone version to Scipion

Once the Standalone version has been installed (see previous section), the user can link such installation to Scipion to have the posibility of use Xmipp inside and outside Scipion. Linking with Scipion requires to the repository of `scipion-em-xmipp` which can be found in the folder `src/scipion-em-xmipp`

This repository contains the files that Scipion needs to execute Xmipp programs. However, it remains to link the Xmipp binaries with Scipion. To do that we need Scipion installed ([see Scipion installation web page](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html#)). With Scipion installed just just launch the next command to link the binaries

`scipion3 installp -p ~/scipion-em-xmipp --devel`

where `scipion-em-xmipp` is the folder of the repository, it means `src/scipion-em-xmipp`.
This command should work in most of the cases. However, if you do this and Scipion does not find Xmipp, you can link Scipion and Xmipp manually by editting the config file of Scipion. This file is located in `scipion/config/scipion.conf`, and it should looks like

```
[PYWORKFLOW]
CONDA_ACTIVATION_CMD = eval "$(/home/username/opt/miniconda3/bin/conda shell.bash hook)"
SCIPION_FONT_SIZE = 6

[PLUGINS]
EM_ROOT = software/em
MAXIT_HOME = %(EM_ROOT)s/maxit-10.1
XMIPP_HOME = /home/username/xmipp-bundle/build
```

The link between Scipion and Xmipp consist in the last line.
```XMIPP_HOME = /home/username/xmipp-bundle/build```. If this line does not exist, it must be added.

# Using Xmipp
Xmipp is installed in the build directory located in the same directory where the xmipp script is located. To set all necessary environment variables and paths to all Xmipp programs, you can simply 
`source build/xmipp.bashrc`


# That's all
Please let us know if you like Xmipp via contacts on the main page of the repository.

If you find a bug or you miss some feature, please create a ticket and we will have a look at it. 
