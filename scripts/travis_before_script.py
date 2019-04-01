#/usr/bin/env python
import os, shutil, subprocess
from distutils.dir_util import copy_tree


def runJob(cmd, cwd='./', show_output=True, log=None, show_command=True,
           inParallel=False):
    if show_command:
        print(green(cmd))
    p = subprocess.Popen(cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    while not inParallel:
        output = p.stdout.readline()
        if output == '' and p.poll() is not None:
            break
        if output:
            l = output.rstrip()
            if show_output:
                print(l)
            if log is not None:
                log.append(l)
    if inParallel:
        return p
    else:
        return 0 == p.poll()


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
            runJob('git mv ' + item_path + ' ' + xmipp_folder)
        else:
            if os.path.isdir(item_path):
                copy_tree(item_path, os.path.join(xmipp_folder, item))
            else:
                shutil.copy(item_path, xmipp_folder)
    except Exception as e:
        print(e)

runJob('git commit -m \'mv files\'')
