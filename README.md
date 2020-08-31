[![Build Status](https://travis-ci.com/I2PC/xmipp.svg?branch=devel)](https://travis-ci.com/I2PC/xmipp)
<!---  [![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=Xmipp&metric=alert_status)](https://sonarcloud.io/dashboard?id=Xmipp)
[![Technical debt](https://sonarcloud.io/api/project_badges/measure?project=Xmipp&metric=sqale_index)](https://sonarcloud.io/component_measures?id=Xmipp&metric=sqale_index)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Xmipp&metric=bugs)](https://sonarcloud.io/project/issues?id=Xmipp&resolved=false&types=BUG)
--->
# Xmipp

Welcome to Xmipp. Xmipp is a suite of image processing programs, primarily aimed at single-particle 3D electron microscopy.


## Notice

Recently we have changed the directory structure. See this wiki [page](https://github.com/I2PC/xmipp/wiki/Transfer-to-new-directory-structure) for more information.

## Getting started
**Xmipp as a Scipion package**

The recommended way to use Xmipp is via [Scipion](https://scipion-em.github.io/docs/index.html).
It can be easily installed using the [Plugin manager](https://scipion-em.github.io/docs/docs/user/plugin-manager.html).

**Xmipp as a standalone bundle (useful for developers)**

Start by cloning the repository from GitHub and go there.
```
git clone https://github.com/I2PC/xmipp xmipp-bundle
cd xmipp-bundle
```

* In case that you want to use/develop it **under Scipion**:
  
  First download the rest of sources by
  ```
  ./xmipp get_devel_sources [branch]
  ```
  where the optional 'branch' parameter will set that given branch.
  
  Secondly, install the 'scipion-em-xmipp' plugin in development mode
  ```
  scipion3 installp -p $PWD/src/scipion-em-xmipp
  ```
  note that 'scipion3' should be installed and visible in the path (check the [Scipion installation guide](https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html)).
  
  Finally, compile Xmipp under Scipion's environ
  ```
  scipion3 installb xmippDev -j 8
  ```
  where `-j 8` indicates that 8 cores will be used to compile Xmipp.
  
* In case you **do NOT want to run Xmipp under Scipion**, just run (it might be necessary to add execute permission via `chmod +x xmipp`)
  ```
  ./xmipp 
  ```

You can see the whole usage of the script with `./xmipp --help`. The most useful options are `br=branch_name` to select a specific branch to be checkout-ed, and `N=#processors` to use for the build (they must be in combination with the `all` option.


---------------


##### In case you want an installation step-by-step. Follow the next receip (also read `./xmipp help`): \
```
git clone https://github.com/I2PC/xmipp xmipp-bundle  # This clones the main Xmipp repo into xmipp-bundle directory
cd xmipp-bundle
./xmipp get_devel_sources [branch]                    # This downloads the rest of Xmipp repos in a certain branch
pip install -e xmipp-bundle/src/scipion-em-xmipp      # this installs Xmipp plugin for Scipion (check the environ/python you are using)
./xmipp config                                        # This configures the Xmipp installation according to the system
./xmipp get_dependencies                              # This downloads the dependencies that Xmipp needs according to the configuration
./xmipp check_config                                  # This checks the configuration set
./xmipp compile [N]                                   # This compiles Xmipp using N processors
./xmipp install [directory]                           # This install Xmipp to a certain directory
```

Please, check [the Xmipp configuration guide](https://github.com/I2PC/xmipp/wiki/Xmipp-configuration-(version-20.07)). 
