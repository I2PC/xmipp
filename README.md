[![Build Status](https://github.com/I2PC/xmipp/actions/workflows/main.yml/badge.svg)](https://github.com/I2PC/xmipp/actions/workflows/main.yml)


# Xmipp

Welcome to Xmipp. Xmipp is a suite of image processing programs, primarily aimed at single-particle 3D electron microscopy.


## Getting started

#### Xmipp as a Scipion package (strongly recommended for non-developers)

The recommended way to use/install Xmipp is via [Scipion](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html).
It can be easily installed using the [Plugin manager](https://scipion-em.github.io/docs/docs/user/plugin-manager.html).

#### Xmipp as a standalone bundle (for developers)

Start by cloning the repository from GitHub and go there.
```
git clone https://github.com/I2PC/xmipp xmipp-bundle
cd xmipp-bundle
```

Please, folow one of the two points below depending on your case. Also, check the [**Xmipp configuration guide**](https://github.com/I2PC/xmipp/wiki/Xmipp-configuration-(version-20.07)). 

* In case that you want to use/develop Xmipp **under Scipion (recommended)**:
  
  First download the rest of sources by
  ```
  ./xmipp get_devel_sources [branch]
  ```
  where the optional 'branch' parameter will set that given git branch (devel by default).
  
  Secondly, install the 'scipion-em-xmipp' plugin in development mode
  ```
  scipion3 installp -p $PWD/src/scipion-em-xmipp --devel
  ```
  note that `scipion3` should be installed and visible in the path (check the [Scipion's installation guide](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html#launching-scipion3)).
  
  Finally, compile Xmipp under Scipion's environ
  ```
  scipion3 installb xmippDev -j 8
  ```
  where `-j 8` indicates that 8 cores will be used to compile Xmipp.
  
  > For a manual compilation of Xmipp, consider to use `scipion3 run ./xmipp [options]`
  
* In case you **do NOT want to get Xmipp under Scipion (only for experts)**, just run
  ```
  ./xmipp 
  ```

  You can see the whole usage of the script with `./xmipp --help`. The most useful options are `br=branch_name` to select a specific branch to be checkout-ed, and `N=#processors` to use for the build (they must be in combination with the `all` option).

---------------


### Detailed installation for Developers.

Follow the next receip (also read `./xmipp help`):
> use `scipion3 run ./xmipp [options]` if you are installing Xmipp under Scipion
```
git clone https://github.com/I2PC/xmipp xmipp-bundle  # This clones the main Xmipp repo into xmipp-bundle directory
cd xmipp-bundle
./xmipp get_devel_sources [branch]                    # This downloads the rest of Xmipp repos in a certain branch
./xmipp config                                        # This configures the Xmipp installation according to the system
./xmipp check_config                                  # This checks the configuration set
./xmipp get_dependencies                              # This downloads the dependencies that Xmipp needs according to the configuration
./xmipp compile [N]                                   # This compiles Xmipp using N processors
./xmipp install [directory]                           # This installs Xmipp to a certain directory

# Optionally, Xmipp plugin for Scipion can be installed under the python/environ of Scipion
pip install -e src/scipion-em-xmipp                   # CHECK the ENVIRON that are present in the session!!
```

_The `./xmipp` and `./xmipp all` commands make the same than the receip above at once (except for the first and last commands) by taking the default values_
