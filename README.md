[![Build Status](https://github.com/I2PC/xmipp/actions/workflows/main.yml/badge.svg)](https://github.com/I2PC/xmipp/actions/workflows/main.yml)

# Xmipp
<a href="https://github.com/I2PC/xmipp"><img src="https://github.com/I2PC/scipion-em-xmipp/blob/devel/xmipp3/xmipp_logo_original.png" alt="drawing" width="330" align="left"></a>


  
Xmipp is a suite of image processing programs, primarily aimed at single-particle 3D electron microscopy, designed and managed by the [Biocomputing Unit](http://biocomputingunit.es/) located in Madrid, Spain.

The Xmipp project is divided into four repositories. 
This is the main repository, which contains the majority of the source code for the programs, additional scripts, and tests. Three remaining repositories, [XmippCore](https://github.com/I2PC/xmippCore/), [XmippViz](https://github.com/I2PC/xmippViz/), and [Scipion-em-xmipp](https://github.com/I2PC/scipion-em-xmipp), are automatically downloaded during the compilation/installation process.


## Getting started
The recommended way for users (not developers) to install and use Xmipp is via the [Scipion](http://scipion.i2pc.es/) framework, where you can use Xmipp with other Cryo-EM-related software. Xmipp  will be installed during Scipion installation (pay attemption on the -noXmipp flag). The Scipion installer should take care of all dependencies for you, however, you can make its life easier if you have your compiler, CUDA and HDF5 library ready and available in the standard paths. Read below for more details about these softwares requirements. Follow the official [installation guide](https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html#installation) of Scipion for more details.

## Installation requirements
### Supported OS
We have tested Xmipp compilation on the following operating systems:
[Ubuntu 16.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-16.04), [Ubuntu 18.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-18.04), [Ubuntu 20.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-20.04), [Ubuntu 22.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-22.04), [Centos 7](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-CentOS-7-9.2009). Visit the OS wiki page for more details.

While compilation and execution might be possible on other systems, it might not be straightforward. If you encounter a problem, please refer to known and fixed [issues](https://github.com/I2PC/xmipp/issues?q=is%3Aissue). Let us know if something is not working.

### Hardware requirements
At least 2 processors are required to run Xmipp. In some virtual machine tools only one is assigned, please check that at least two processors are assigned to the virtual machine

### Software dependencies
#### Compiler
Xmipp requires C++17 compatible compiler. We recommend either GCC or CLANG, in the newest version possible. We have good experience with GCC-8, in any case a version > 8 is required. If use GCC-11 and experience issues, [please visit this.](https://github.com/I2PC/xmipp/issues/583). For more details about the compilation proces and installation of gcc, please visit [compiler](https://github.com/I2PC/xmipp/wiki/Compiler)

#### Cmake
Xmipp requires Cmake 3.16 or above. To update it please [visit](https://github.com/I2PC/xmipp/wiki/Cmake-update-and-install)
#### Cuda
Xmipp supports Cuda 8 through 11.7. CUDA is optional but highly recommended. We recommend you to use the newest version available for your operating system, though Cuda 10.2 has the widest support among other Scipion plugins. Pay attemption to the [compiler - CUDA](https://gist.github.com/ax3l/9489132) compatibility.
To install CUDA for your operating system, follow the [official install guide](https://developer.nvidia.com/cuda-toolkit-archive).

#### OpenCV
OpenCV is used for some programs: movie_optical_alignment (with GPU support) and volume_homogenizer, however, it is not required.
If you installed OpenCV via apt (`sudo apt install libopencv-dev`), it should be automatically picked up by the Xmipp script

#### HDF5
We sometimes see issues regarding the HDF5 dependency. Please visit the [HDF5 Troubleshooting wiki](https://github.com/I2PC/xmipp/wiki/HDF5-Troubleshooting)

#### Full list of dependencies

`sudo apt install -y libfftw3-dev libopenmpi-dev libhdf5-dev python3-numpy python3-dev libtiff5-dev libsqlite3-dev default-jdk git cmake`

`pip install scons` in the enviroment that xmipp will be compiled (scipion3 if you will run Xmipp within Scipion)

Also a compiler will be required (`sudo apt install gcc-8 g++-8`)

## Standalone installation
Standalone installation of Xmipp is recommended for researchers and developers. This installation allows you to use Xmipp without Scipion, however, in the next section it is explained how to link it with Scipion. Xmipp script automatically downloads several dependencies and then creates a configuration file that contains paths and flags used during the compilation. Please refer to the [Xmipp configuration](https://github.com/I2PC/xmipp/wiki/Xmipp-configuration) guide for more info.

Start by cloning the repository and then navigate to the right directory.

`git clone https://github.com/I2PC/xmipp.git xmipp-bundle && cd xmipp-bundle`

Refer to `./xmipp --help` for additional info on the compilation process and possible customizations.

Next is to compile xmipp. There are to possibilities and in both you will can run Xmipp in Scipion (see [linking step](https://github.com/I2PC/xmipp/edit/agm_refactoring_readme/README.md#linking-xmipp-standalone-to-scipion))
- Compile Xmipp by invoking the compilation script, which will take you through the rest of the process:`./xmipp` that way you will install Xmipp with the dependencies and thier versions that the enviroment you decide, or the default one.
- Compile Xmipp via Scipion enviroment `scipion3 run ./xmipp` that way you will install Xmipp with the dependencies and their versions that Scipion decided. 

It is important to highlight that this step only compiles Xmipp, but it does not link to Scipion. The linking to Scipion is explained in the next section.

### Linking Xmipp standalone to Scipion

Once the Standalone version has been installed, the user can link such installation to Scipion to have the posibility of use Xmipp inside Scipion. Linking with Scipion requires to the repository of `scipion-em-xmipp` which can be found in the folder `src/scipion-em-xmipp`. This repository contains the files that Scipion needs to execute Xmipp programs. However, it remains to link the Xmipp binaries with Scipion. To do that we need Scipion installed ([see Scipion installation web page](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html#)) and just launch the next command to link the binaries

`scipion3 installp -p ~/scipion-em-xmipp --devel`

where `scipion-em-xmipp` is the folder of the repository, it means `src/scipion-em-xmipp`.
This command should work in most of the cases. However, if you do this and Scipion does not find Xmipp, please visit [Linking Xmipp to Scipion Troubleshooting](https://github.com/I2PC/xmipp/wiki/Linking-Xmipp-to-Scipion-Troubleshooting)

## Using Xmipp
Xmipp is installed in the build directory located in the same directory where the xmipp script is located. To set all necessary environment variables and paths to all Xmipp programs, you can simply 
`source build/xmipp.bashrc`


## That's all

If you miss some feature or find a bug, please [create an issue](https://github.com/I2PC/xmipp/issues/new) and we will have a look at it, you can also contact us by xmipp@cnb.csic.es. For more details, troubleshootings and information visit the [wikip page.](https://github.com/I2PC/xmipp/wiki)

If you like this project, Watch or Star it! We are also looking for new collaborators, so contact us if you’re interested
