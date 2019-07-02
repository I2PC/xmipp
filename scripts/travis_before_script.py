#/usr/bin/env python
import os, shutil
from distutils.dir_util import copy_tree

folder = './'
src_folder_name = 'src'
xmipp_folder_name = 'xmipp'
xmipp_folder = os.path.join(folder, src_folder_name, xmipp_folder_name)
xmipp_script = 'xmipp'
sonar_script = 'sonar-project.properties'
sync_script = 'sync_data.py'
git_folder = '.git'
# get folder relative to home dir and strip os.path separators
script_dir = os.path.dirname(os.path.realpath(__file__)).split(os.getcwd())[1][1:]
copy_list = [xmipp_script, script_dir, sonar_script, sync_script]
if 'TRAVIS' in os.environ:
    # we need git folder so that SonarCloud can use it for PR decoration
    copy_list.append(git_folder)
black_list = [src_folder_name]
os.makedirs(xmipp_folder)
for item in os.listdir(folder):
    if item in black_list:
        continue
    item_path = os.path.join(folder, item)
    try:
        if item not in copy_list:
            shutil.move(item_path, xmipp_folder)
        else:
            if os.path.isdir(item_path):
                copy_tree(item_path, os.path.join(xmipp_folder, item))
            else:
                shutil.copy(item_path, xmipp_folder)
    except Exception as e:
        print(e)

