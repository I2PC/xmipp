[![Build Status](https://github.com/I2PC/xmipp/actions/workflows/build.yml/badge.svg)](https://github.com/I2PC/xmipp/actions/workflows/build.yml)

# Xmipp
<a href="https://github.com/I2PC/xmipp"><img src="https://github.com/I2PC/scipion-em-xmipp/blob/devel/xmipp3/xmipp_logo.png" alt="drawing" width="330" align="left"></a>


  
Xmipp is a suite of image processing programs, primarily aimed at single-particle 3D electron microscopy, designed and managed by the [Biocomputing Unit](http://biocomputingunit.es/) located in Madrid, Spain.

The Xmipp project is divided into four repositories. 
This is the main repository, which contains the majority of the source code for the programs, additional scripts, and tests. Three remaining repositories, [XmippCore](https://github.com/I2PC/xmippCore/), [XmippViz](https://github.com/I2PC/xmippViz/), and [Scipion-em-xmipp](https://github.com/I2PC/scipion-em-xmipp), are automatically downloaded during the installation process.


## Getting started
To have a complete overview about Xmipp please visit the [documentation web](https://i2pc.github.io/docs/). The recommended way for users (not developers) to install and use Xmipp is via the [Scipion](http://scipion.i2pc.es/) framework, where you can use Xmipp with other Cryo-EM-related software. 

### As a Scipion plugin (Recommended)
The Scipion installer should take care of all dependencies for you by default, except for CUDA. To correctly interoperate with CUDA, please visit the official [installation guide](https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html#installation) of Scipion.

Once Scipion is in working order, launch the Xmipp Plugin installer with

`scipion3 installp -p scipion-em-xmipp`

By default this will employ all CPU cores available in the system for compilation. If this is not the desired behavior, please consider using `-j <N>` flag to limit the amount of CPU cores.

### As a Scipion plugin with pre-installed dependencies
The former method also installs dependencies (including the compiler) though conda. If you prefer to use externally installed dependencies and avoid downloading them from conda use

`scipion3 installp -p scipion-em-xmipp --noBin`

`scipion3 installb xmippSrc`

Mind that for this command to succeed, all dependencies need to be pre-installed and findable by Xmipp. When using non-standard install directories, this needs to be indicated to Xmipp though the Scipion configuration file. See Software dependency section for further details.

### Developer installation with Scipion
For developers, it becomes handy to install Xmipp in an external directory for easy integration with IDEs. By default Xmipp installer takes care of installing dependencies though conda and linking to Scipion.

The first step is to download Xmipp sources:

`git clone https://github.com/I2PC/xmipp.git`

`cd xmipp`

Then all child repositories need to be fetched, with an optional git branch parameter. For repositories where the branch does not exist, devel is utilized.

`./xmipp getSources [-b branch]`

Install the Scipion plugin and dependencies

`scipion3 installp -p src/scipion-em-xmipp --devel`

Compile Xmipp

`scipion3 run ./xmipp`

Refer to `./xmipp --help` for additional info on the compilation process and possible customizations.

#### Integrating with Visual Studio Code

## Using Xmipp
Xmipp is installed in the build directory located in the same directory where the xmipp script is located. To set all necessary environment variables and paths to all Xmipp programs, you can simply 
`source dist/xmipp.bashrc`

## Installation requirements
### Supported OS
We have tested Xmipp compilation on the following operating systems:
[Ubuntu 18.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-18.04), [Ubuntu 20.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-20.04), [Ubuntu 22.04](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-Ubuntu-22.04), [Centos 7](https://github.com/I2PC/xmipp/wiki/Installing-Xmipp-on-CentOS-7-9.2009). Visit the OS wiki page for more details.

While compilation and execution might be possible on other systems, it might not be straightforward. If you encounter a problem, please refer to known and fixed [issues](https://github.com/I2PC/xmipp/issues?q=is%3Aissue). Let us know if something is not working.

### Hardware requirements
At least 2 processors are required to run Xmipp. In some virtual machine tools only one is assigned, please check that at least two processors are assigned to the virtual machine

### Software dependencies
Note that these dependencies are installed by default though conda (except for CUDA). Only install them externally when required.

#### Compiler
Xmipp requires C++17 compatible compiler (see table below for minimum versions). We have had good experience with using GCC. Clang is mostly untested. Mind that when compiling with CUDA, a [compatible compiler](https://gist.github.com/ax3l/9489132) must be installed and findable (although it may not be used for non-CUDA sources) for Xmipp.

| Compiler | Minumum version |
|----------|-----------------|
| GCC      | 8.4             |
| Clang    | 5.0             |


#### CMake
Xmipp requires CMake 3.17 or above. To update it please [visit](https://github.com/I2PC/xmipp/wiki/Cmake-update-and-install)

#### CUDA
Xmipp supports CUDA 8 through 11.7. CUDA is optional but highly recommended. We recommend you to use the newest version available for your operating system. Some Xmipp programs are only compiled if CUDA 11 is available. Pay attention to the [compiler - CUDA](https://gist.github.com/ax3l/9489132) compatibility.

To install CUDA for your operating system, follow the [official install guide](https://developer.nvidia.com/cuda-toolkit-archive).

#### Installing dependencies via apt
`sudo apt install -y libfftw3-dev libopenmpi-dev libhdf5-dev python3-numpy python3-dev libtiff5-dev libsqlite3-dev default-jdk git cmake gcc-10 g++-10`


## That's all

If you miss some feature or find a bug, please [create an issue](https://github.com/I2PC/xmipp/issues/new) and we will have a look at it, you can also contact us by xmipp@cnb.csic.es. For more details, troubleshootings and information visit the [wiki page.](https://github.com/I2PC/xmipp/wiki)

If you like this project, Watch or Star it! We are also looking for new collaborators, so contact us if you’re interested
