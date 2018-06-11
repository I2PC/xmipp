#!/usr/bin/env python

# **************************************************************************
# *
# * Authors:     I. Foche Perez (ifoche@cnb.csic.es)
# *              J.M. de la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *
# * Unidad de Bioinformatica of Centro Nacional de Biotecnologia, CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'ifoche@cnb.csic.es'
# *
# **************************************************************************

import os
from os.path import join
from glob import glob
from datetime import datetime


Import('env')


AddOption('--no-opencv', dest='opencv', action='store_false', default=True,
          help='Avoid compilation of opencv programs')


# Define some variables used by Scons. Note that some of
# the variables will be passed by Scipion in the environment (env).

env['CUDA_SDK_PATH'] = os.environ.get('CUDA_SDK_PATH', '')
env['CUDA_LIB_PATH'] = os.environ.get('CUDA_LIB_PATH', '')

get = lambda x: os.environ.get(x, '0').lower() in ['true', 'yes', 'y', '1']

gtest = get('GTEST')
cuda = get('CUDA')
debug = get('DEBUG')
matlab = get('MATLAB')
opencv = env.GetOption('opencv') and get('OPENCV')

if opencv:
    opencvLibs = ['opencv_core',
                  # 'opencv_legacy',
                  'opencv_imgproc',
                  'opencv_video',
                  'libopencv_calib3d']
                  
else:
    opencvLibs = []



if 'MATLAB' in os.environ:
    # Must be removed to avoid problems in Matlab compilation.
    del os.environ['MATLAB']
    # Yeah, nice

# Read some flags
CYGWIN = env['PLATFORM'] == 'cygwin'
MACOSX = env['PLATFORM'] == 'darwin'
MINGW = env['PLATFORM'] == 'win32'

XMIPP_PATH = Dir('.').abspath


#  ***********************************************************************
#  *                      Xmipp C++ Libraries                            *
#  ***********************************************************************

ALL_LIBS = {'fftw3', 'tiff', 'jpeg', 'sqlite3', 'hdf5'}

# Create a shortcut and customized function
# to add the Xmipp CPP libraries
def addLib(name, **kwargs):
    ALL_LIBS.add(name)
    # Install all libraries in scipion/software/lib
    # COSS kwargs['installDir'] = '#software/lib'
    # Add always the xmipp path as -I for include and also xmipp/libraries
    incs = kwargs.get('incs', []) + [join(XMIPP_PATH, 'external'),
                                     join(XMIPP_PATH, 'libraries'), join(env['XMIPP_BUNDLE'],'xmippCore')]
    kwargs['incs'] = incs

    deps = kwargs.get('deps', [])
    kwargs['deps'] = deps

    # Add libraries in libs as deps if not present
    libs = kwargs.get('libs', [])
    for lib in libs:
        if lib in ALL_LIBS and lib not in deps:
            deps.append(lib)

    # If pattern not provided use *.cpp as default
    patterns = kwargs.get('patterns', '*.cpp')
    kwargs['patterns'] = patterns
    
    if 'cuda' in kwargs and kwargs['cuda']:
    	lib = env.AddCppLibraryCuda(name, **kwargs)
    else:    	
    	lib = env.AddCppLibrary(name, **kwargs)
    	
    env.Alias('xmipp-libs', lib)

    return lib


# Add first external libraries (alglib, bilib, condor)
#NOTE: for alglib and condor, the dir can not be where the source
# code is because try to use .h files to make the final .so library

addLib('XmippExternal',
       dirs=['external','external','external'],
       patterns=['condor/*.cpp','delaunay/*.cpp','gtest/*.cc'],
       libs=['pthread'])

EXT_LIBS = ['XmippExternal']
env.Alias('XmippExternal', EXT_LIBS)


# Data
#TODO: checklib rt?????
addLib('XmippData',
       dirs=['libraries'],
       patterns=['data/*.cpp'])

# Classification
addLib('XmippClassif',
       dirs=['libraries'],
       patterns=['classification/*.cpp'],
       libs=['XmippData'])

# Dimred
addLib('XmippDimred',
       dirs=['libraries'],
       patterns=['dimred/*.cpp'],
       libs=['XmippData'])

# Reconstruction
addLib('XmippRecons',
       dirs=['libraries'],
       patterns=['reconstruction/*.cpp'],
       libs=['XmippData', 'XmippClassif', 'XmippExternal'])


# Reconstruction CUDA
if cuda:
    addLib('XmippReconsCuda',
       dirs=['libraries'],
       patterns=['reconstruction_cuda/*.cpp'],
       incs=bilib_incs, cuda=True,
       libs=['pthread'])
       
    addLib('XmippReconsAdaptCuda',
       dirs=['libraries'],
       patterns=['reconstruction_adapt_cuda/*.cpp'],
       incs=bilib_incs,
       libs=['pthread'])
       
    addLib('XmippParallelAdaptCuda',
       dirs=['libraries'],
       patterns=['parallel_adapt_cuda/*.cpp'],
       incs=bilib_incs,
       libs=['pthread', 'XmippData', 'XmippClassif', 'XmippRecons', 'XmippExternal', 'XmippReconsCuda', 'XmippReconsAdaptCuda'],
       mpi=True)


# Interface
def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]
env['PYTHONINCFLAGS'] = os.environ.get('PYTHONINCFLAGS', '').split()
if len(env["PYTHONINCFLAGS"])>0:
    python_incdirs = [remove_prefix(os.path.expandvars(x),"-I") for x in env["PYTHONINCFLAGS"]]
else:
    python_incdirs = []

addLib('XmippInterface',
       dirs=['libraries'],
       patterns=['interface/*.cpp'],
       incs=python_incdirs,
       libs=[ 'pthread', 'python2.7',
             'XmippData', 'XmippBilib'])

# Parallelization
addLib('XmippParallel',
              dirs=['libraries'],
              patterns=['parallel/*.cpp'],
              libs=['pthread', 'fftw3_threads',
                    'XmippData', 'XmippClassif', 'XmippRecons'],
              mpi=True)

# Python binding
addLib('xmipp.so',
       dirs=['libraries/bindings'],
       patterns=['python/*.cpp'],
       incs=python_incdirs,
       libs=['python2.7', 'XmippData', 'XmippRecons'],
       prefix='', target='xmipp')


#  ***********************************************************************
#  *                      Xmipp Programs and Tests                       *
#  ***********************************************************************

XMIPP_LIBS = ['XmippData', 'XmippRecons', 'XmippClassif']
PROG_DEPS = EXT_LIBS + XMIPP_LIBS

PROG_LIBS = EXT_LIBS + XMIPP_LIBS

def addRunTest(testName, prog):
    """ Add a Scons target for running xmipp tests. """
    xmippTestName = 'xmipp_' + testName
    xmlFileName = join(XMIPP_PATH, 'applications', 'tests', 'OUTPUT',
                       xmippTestName+".xml")
    if os.path.exists(xmlFileName):
        os.remove(xmlFileName)
    testCase = env.Command(
        xmlFileName,
        join(XMIPP_PATH, 'bin/%s' % xmippTestName),
        "%s/scipion run $SOURCE --gtest_output=xml:$TARGET" % os.environ['SCIPION_HOME'])
    env.Alias('run_' + testName, testCase)
    env.Depends(testCase, prog)
    env.Alias('xmipp-runtests', testCase)

    AlwaysBuild(testCase)

    return testCase


# Shortcut function to add the Xmipp programs.
def addProg(progName, **kwargs):
    """ Shortcut to add the Xmipp programs easily.
    Params:
        progName: the name of the program without xmipp_ prefix that will be added.
        if 'src' not in kwargs, add: 'applications/programs/progName' by default.
        if progName starts with 'mpi_' then mpi will be set to True.
    """
    isTest = progName.startswith('test_')

    progsFolder = 'tests' if isTest else 'programs'

    src = kwargs.get('src', [join('applications', progsFolder, progName)])

    kwargs['src'] = src
    # Add all xmipp libraries just in case
    kwargs['libs'] = kwargs.get('libs', []) + PROG_LIBS
    kwargs['deps'] = PROG_DEPS

    # Add always the xmipp path as -I for include and also xmipp/libraries
    incs = kwargs.get('incs', []) + [join(XMIPP_PATH, 'external'),
                                     join(XMIPP_PATH, 'libraries')]
    kwargs['incs'] = incs

    if progName.startswith('mpi_'):
        kwargs['mpi'] = True
        kwargs['libs'] += ['mpi', 'mpi_cxx', 'XmippParallel']

    #AJ
    if progName.startswith('cuda_'):
        kwargs['cuda'] = True
        #kwargs['nvcc'] = True
        kwargs['libs'] += ['XmippReconsAdaptCuda', 'XmippReconsCuda']

    if progName.startswith('mpi_cuda_'):
        kwargs['cuda'] = True
        #kwargs['nvcc'] = True
        kwargs['libs'] += ['XmippReconsAdaptCuda', 'XmippReconsCuda','XmippParallelAdaptCuda']
    #FIN AJ

    xmippProgName = 'xmipp_%s' % progName

    if progName.startswith('test_'):
        kwargs['libs'] += ['XmippGtest']
        env.Alias('xmipp-tests', xmippProgName)
        addRunTest(progName, xmippProgName)

    prog = env.AddProgram(xmippProgName, **kwargs)

    # Add some aliases before return
    env.Alias(xmippProgName, prog)
    env.Alias('xmipp-programs', prog)


    return prog


# Define the list of all programs
for p in ['angular_break_symmetry',
          'angular_commonline',
          'angular_continuous_assign',
          'angular_continuous_assign2',
          'angular_accuracy_pca',
          'angular_discrete_assign',
          'angular_distance',
          'angular_distribution_show',
          'angular_estimate_tilt_axis',
          'angular_neighbourhood',
          'angular_projection_matching',
          'angular_project_library',
          'angular_rotate',

          'classify_analyze_cluster',
          'classify_compare_classes',
          'classify_evaluate_classes',
          'classify_extract_features',
          'classify_first_split',
          'classify_kerdensom',
          'coordinates_noisy_zones_filter',
	        'ctf_correct_wiener2d',
          'ctf_correct_wiener3d',
          'ctf_correct_idr',
          'ctf_create_ctfdat',
          'ctf_enhance_psd',
          'ctf_estimate_from_micrograph',
          'ctf_estimate_from_psd',
          'ctf_estimate_from_psd_fast',
          'ctf_group',
          'ctf_phase_flip',
          'ctf_show',
          'ctf_sort_psds',

          'evaluate_coordinates',

          'idr_xray_tomo',

          'flexible_alignment',
          'image_align',
          'image_align_tilt_pairs',
          'image_assignment_tilt_pair',
          'image_common_lines',
          'image_convert',
          'image_eliminate_empty_particles',
          'image_find_center',
          'image_header',
          'image_histogram',
          'image_operate',
          'image_rotational_pca',
          'image_residuals',
          'image_resize',
          'image_rotational_spectra',
          'image_separate_objects',
          'image_sort_by_statistics',
          'image_ssnr',
          'image_statistics',
          'image_vectorize',

          ('matrix_dimred', ['XmippDimred']),
          #'metadata_convert_to_spider',
          'metadata_histogram',
          'metadata_import',
          'metadata_split',
          'metadata_split_3D',
          'metadata_utilities',
          'metadata_xml',
          'micrograph_scissor',
          'micrograph_automatic_picking',
          'ml_align2d',
          'mlf_align2d',
          'ml_refine3d',
          'mlf_refine3d',
          'ml_tomo',
          'movie_alignment_correlation',
          'movie_filter_dose',
          'movie_estimate_gain',
          'mrc_create_metadata',
          'multireference_aligneability',
          'nma_alignment',
          'nma_alignment_vol',

          'pdb_analysis',
          'pdb_construct_dictionary',
          'pdb_nma_deform',
          'pdb_restore_with_dictionary',
          'pdb_reduce_pseudoatoms',
          'phantom_create',
          'phantom_project',
          'phantom_simulate_microscope',
          'phantom_transform',

          'reconstruct_admm',
          'reconstruct_art',
          'reconstruct_art_pseudo',
          'reconstruct_art_xray',
          'reconstruct_fourier',
          'reconstruct_fourier_accel',
          'reconstruct_significant',
          'reconstruct_wbp',
          'resolution_fsc',
          'resolution_ibw',
          'resolution_monogenic_signal',
          'resolution_ssnr',
          'score_micrograph',

          'work_test',

          'transform_add_noise',
	        'transform_adjust_image_grey_levels',
          'transform_adjust_volume_grey_levels',
          'transform_center_image',
          ('transform_dimred', ['XmippDimred']),
          'transform_downsample',
          'transform_filter',
          'transform_geometry',
          'transform_mask',
          'transform_mirror',
          'transform_morphology',
          'transform_normalize',
          'transform_randomize_phases',
          'transform_range_adjust',
          'transform_symmetrize',
	      'transform_threshold',
          'transform_window',
          'tomo_align_dual_tilt_series',
          'tomo_align_refinement',

          'tomo_align_tilt_series',
          'tomo_detect_missing_wedge',
          'tomo_project',
          'tomo_remove_fluctuations',
          'tomo_extract_subvolume',
          'validation_nontilt',
          'validation_tilt_pairs',

          'volume_center',
          'volume_correct_bfactor',
          'volume_enhance_contrast',
          'volume_find_symmetry',
          'volume_from_pdb',
          'volume_halves_restoration',
          'volume_initial_simulated_annealing',
          'volume_validate_pca',
          'volume_pca',
          'volume_reslice',
          'volume_segment',
          'volume_structure_factor',
          'volume_to_pseudoatoms',
          'volume_to_web',

          'xray_import',
          'xray_psf_create',
          'xray_project',

          # MPI programs, the mpi_ prefix set mpi=True flag.
          'mpi_angular_class_average',
          'mpi_angular_continuous_assign',
          'mpi_angular_continuous_assign2',
          'mpi_angular_accuracy_pca',
          'mpi_angular_discrete_assign',
          'mpi_angular_projection_matching',
          'mpi_angular_project_library',
          'mpi_classify_CL2D',
          'mpi_classify_CL2D_core_analysis',
          'mpi_ctf_correct_idr',
          'mpi_ctf_sort_psds',
          'mpi_ctf_correct_wiener2d',
          'mpi_image_operate',
          'mpi_image_rotational_pca',
          'mpi_image_resize',
          'mpi_image_sort',
          'mpi_image_ssnr',
          'mpi_ml_align2d',
          'mpi_ml_tomo',
          'mpi_mlf_align2d',
          'mpi_ml_refine3d',
          'mpi_mlf_refine3d',
          'mpi_multireference_aligneability',
          'mpi_nma_alignment',
          'mpi_performance_test',
          'mpi_reconstruct_art',
          'mpi_reconstruct_fourier',
          'mpi_reconstruct_fourier_accel',
          'mpi_reconstruct_wbp',
          'mpi_reconstruct_significant',
          'mpi_reconstruct_admm',
          'mpi_run',
          'mpi_tomo_extract_subvolume',
	  'mpi_transform_adjust_image_grey_levels',
          'mpi_transform_filter',
          'mpi_transform_symmetrize',
          'mpi_transform_geometry',
          'mpi_transform_mask',
          'mpi_transform_normalize',
          'mpi_transform_threshold',
          'mpi_xray_project',
	      'mpi_validation_nontilt',	  
          'mpi_write_test',

          # Unittest for Xmipp libraries
          'test_ctf',
          ('test_dimred', ['XmippDimred']),
          'test_euler',
          'test_fftw',
          'test_filename',
          'test_filters',
          'test_fringe_processing',
          'test_funcs',
          'test_geometry',
          'test_image',
          'test_image_generic',
          'test_matrix',
          'test_metadata',
          'test_movie_filter_dose',
          'test_multidim',
          'test_polar',
          'test_polynomials',
          'test_resolution_frc',
          'test_sampling',
          'test_symmetries',
          'test_transformation',
          'test_transform_window',
          'test_wavelets'
          ]:
    # Allow to add specific libs for indiviual programs
    # by using a tuple instead of string
    if isinstance(p, tuple):
        addProg(p[0], libs=p[1])
    else:
        addProg(p)


# Programs with specials needs
# This programs need python lib to compile
addProg('volume_align_prog',
         incs=python_incdirs,
         libs=['python2.7', 'XmippInterface'])
addProg('mpi_classify_CLTomo_prog',
         incs=python_incdirs,
         libs=['python2.7', 'XmippInterface'])

# Optical Alignment program, that depends on opencv (cpu version)
if opencv:
    addProg('movie_optical_alignment_cpu',
            libs=opencvLibs,
            deps=['opencv'])
    addProg('mpi_volume_homogenizer',
            libs=opencvLibs,
            deps=['opencv'])
    if cuda: # also the gpu version
        opencvLibs.append('opencv_gpu')
        addProg('movie_optical_alignment_gpu',
                libs=opencvLibs,
                deps=['opencv'], cuda=True)


if cuda:
    addProg('cuda_correlation')
    addProg('cuda_reconstruct_fourier')
    addProg('mpi_cuda_reconstruct_fourier')


#  ***********************************************************************
#  *                      Xmipp Scripts                                  *
#  ***********************************************************************


def addBatch(batchName, script, scriptFolder='applications/scripts'):
    """ Add a link to xmipp/bin folder prepending xmipp_ prefix.
    The script should be located in from xmipp root,
    by default in 'applications/scripts/'
    """
    xmippBatchName = 'xmipp_%s' % batchName
    batchLink = env.SymLink(join(XMIPP_PATH, 'bin', xmippBatchName),
                            join(XMIPP_PATH, scriptFolder, script))
    env.Alias('xmipp-batchs', batchLink)

    return batchLink


# Batches (apps)

addBatch('apropos', 'apropos/batch_apropos.py')
addBatch('compile', 'compile/batch_compile.py')
addBatch('metadata_selfile_create', 'metadata_selfile_create/batch_metadata_selfile_create.py')
addBatch('volume_align', 'volume_align/batch_volume_align.sh')
addBatch('mpi_classify_CLTomo', 'mpi_classify_CLTomo/batch_mpi_classify_CLTomo.sh')

# # Python tests
# testPythonInterface = env.SymLink('bin/xmipp_test_pythoninterface', 'applications/tests/test_pythoninterface/batch_test_pythoninterface.py')
# Depends(testPythonInterface, packageDeps)
# AddXmippTest('test_pythoninterface', testPythonInterface, "$SOURCE $TARGET")
#
# testPySqlite = env.SymLink('bin/xmipp_test_pysqlite', 'applications/tests/test_pysqlite/batch_test_pysqlite.py')
# Depends(testPySqlite, packageDeps)
# AddXmippTest('test_pysqlite', testPySqlite, "$SOURCE $TARGET")
#
# testEMX = env.SymLink('bin/xmipp_test_emx', 'applications/tests/test_emx/batch_test_emx.py')
# Depends(testEMX, packageDeps)
# AddXmippTest('test_emx', testEMX, "$SOURCE $TARGET")


def compileMatlabBinding(target, source, env):
    matlabDir = join(XMIPP_PATH, 'libraries', 'bindings', 'matlab')

    incStr = ' '.join('-I%s' % p for p in [os.path.join(XMIPP_PATH, 'libraries'),
                                            os.path.join(XMIPP_PATH, 'external')]+env['EXTERNAL_INCDIRS'] )
    libStr = ' '.join('-L%s' % p for p in env['EXTERNAL_LIBDIRS'])

    libs = ' '.join('-l%s' % lib for lib in ['XmippData',
                                             'XmippRecons',
                                             'XmippBilib',
                                             'XmippAlglib'])

    mex = join(env['MATLAB_DIR'], 'bin', 'mex')
    command = '%s -O CFLAGS="\$CFLAGS -std=c99 -fpermissive" -outdir %s %s %s %s %s ' % (mex, matlabDir, incStr, libStr, libs, source[0])
    print command
    os.system(command)

# Matlab programs
def addMatlabBinding(name):
    """ Add options to compile xmipp-Matlab bindings. """
    matlabDir = join(XMIPP_PATH, 'libraries', 'bindings', 'matlab')
    source = join(matlabDir, name + ".cpp")
    target = join(matlabDir, name + ".mexa64")

    cmdTarget = env.Command(target, source, compileMatlabBinding)
    env.Alias('xmipp-matlab', cmdTarget)

    return cmdTarget

if matlab:
    bindings = [#'tom_xmipp_adjust_ctf',
                'tom_xmipp_align2d',
                'tom_xmipp_ctf_correct_phase',
                'tom_xmipp_mask',
                'tom_xmipp_mirror',
                'tom_xmipp_morphology',
                'tom_xmipp_normalize',
                'tom_xmipp_psd_enhance',
                'tom_xmipp_resolution',
                'tom_xmipp_rotate',
                'tom_xmipp_scale',
                'tom_xmipp_scale_pyramid',
                'tom_xmipp_volume_segment',

                'mirt3D_mexinterp',

                'xmipp_ctf_generate_filter',
                'xmipp_nma_read_alignment',
                'xmipp_nma_save_cluster',
                'xmipp_read',
                'xmipp_read_structure_factor',
                'xmipp_write',
                ]
    for b in bindings:
        addMatlabBinding(b)

    env.Default('xmipp-matlab')
    env.Alias('xmipp', 'xmipp-matlab')

XmippAlias = env.Alias('xmipp', ['xmipp-libs',
                                 'xmipp-programs',
                                 'xmipp-batchs'])


Return('XmippAlias')
