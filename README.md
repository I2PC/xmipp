[![Build Status](https://travis-ci.com/I2PC/xmipp.svg?branch=devel)](https://travis-ci.com/I2PC/xmipp)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=Xmipp&metric=alert_status)](https://sonarcloud.io/dashboard?id=Xmipp)
[![Technical debt](https://sonarcloud.io/api/project_badges/measure?project=Xmipp&metric=sqale_index)](https://sonarcloud.io/component_measures?id=Xmipp&metric=sqale_index)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Xmipp&metric=bugs)](https://sonarcloud.io/project/issues?id=Xmipp&resolved=false&types=BUG)

# xmipp

**If you want to use it as a Scipion package**, please visit [this](https://github.com/I2PC/xmipp/wiki/Migrating-branches-from-nonPluginized-Scipion-to-the-new-Scipion-Xmipp-structure#xmipp-plugin).

To install Xmipp in a certain place (e.g. in the `xmipp-bundle` directory) follow this instructions:

```
mkdir xmipp-bundle
cd xmipp-bundle
wget https://raw.githubusercontent.com/I2PC/xmipp/devel/xmipp -O xmipp
chmod 755 xmipp
./xmipp all N=4 br=devel
ln -sf src/xmipp/xmipp xmipp  # optional, but VERY RECOMMENDED, to have always the last version of the xmipp script
```
where you can replace `N=4` for `N=#processors` and `br=master` for `br=devel` if you want the development version of Xmipp. You can see the whole usage of the xmipp script with `./xmipp --help`


### Using Scipion libraries

If you want xmipp to pick up SCIPION libraries define SCIPION_HOME=\<path to scipion\>
