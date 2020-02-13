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

In case you don't want to import settings from Scipion, set XMIPP_NOSCIPION flag to true (`export XMIPP_NOSCIPION=True`). 

Otherwise, run `xmipp` script in the root folder via Scipion (it might be necessary to add execute permission via `chmod +x xmipp`)
```
/<path to scipion>/scipion run ./xmipp
```
Running the script through Scipion will properly set the enviroment. This script will checkout additional repositories and build Xmipp for you.

You can see the whole usage of the script with `./xmipp --help`. The most useful options are `br=branch_name` to select a specific branch to be checkout-ed, and `N=#processors` to use for the build.


## FAQ

If you want to use your specific version of Xmipp as a Scipion plugin, see following wiki [page](https://github.com/I2PC/xmipp/wiki/Migrating-branches-from-nonPluginized-Scipion-to-the-new-Scipion-Xmipp-structure#xmipp-plugin).
