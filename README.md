# xmipp

To install Xmipp follow this instructions:

```
mkdir <xmipp-bundle>
cd <xmipp-bundle>
wget https://raw.githubusercontent.com/I2PC/xmipp/devel/xmipp -O xmipp
chmod 755 xmipp
./xmipp [all N]
ln -sf src/xmipp/xmipp xmipp  # optional but recommended to have always the last version of the xmipp script
```
where N (8 by default) is the number of processors that you want to use for compile.

If you want to use it as a Scipion package, please visit [this](https://github.com/I2PC/xmipp/wiki/Migrating-branches-from-nonPluginized-Scipion-to-the-new-Scipion-Xmipp-structure#xmipp-software).
