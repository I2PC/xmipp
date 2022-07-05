  ## Release 3.22.XX - devel
  ### Installation and user guide
  -

  ### Scripts Xmipp
  -

  ### Protocols scipion-em-xmipp
  -

  ### Others
  -


## Release 3.22.07 - Helios

### Scripts Xmipp
- **xmipp_image_operate** taked into account non existing files
- **angular_continuous_assign2** Bug fixed
- **volume_consensus** Bug fixed
- **ctf.h and angular_continuous_assign_2** changes for local defocus estimation #578

### Protocols scipion-em-xmipp
- **protocol_rotate_volume.py** New protocol
- **protocol_subtract_projection**: New implementation based on adjustment by regression instead of POCS and improved performance
- **protocol_local_ctf** add new sameDefocus option + formatting
- **protocol_compare_reprojections.py & protocol_align_volume.py** Fast Fourier by default
- **protocol_crop_resize.py** allows input pointers
- **protocol_resolution_deepres.py** resize output to original size
- **protocol_denoise_particles.py** Added setOfAverages as input option
- **protocol_process.py** change output from stk (spider) to mrcs (mrc)
- **protocol_trigger_data.py** Bug fixed
- **protocol_screen_deeplearning**  Added descriptive help" 
- **protocol_center_particles.py** Added summary info
- **New tests:** test_protocol_reconstruct_fourier, test_protocols_local_defocus.py, test_protocols_local_defocus**
- **Improved tests:** test_protocols_deepVolPostprocessing, test_protocols_xmipp_3d, Test ProjSubtracion
- **Excluded tests:** test_protocols_zernike3d, test_protocols_metaprotocol_heterogeneity


### Installation and user guide
- Removed empty folder with cleanBin option
- Clarifing linking to Scipion and removed a bug with the build link
- New flag (OPENCV_VERSION) in xmipp.config
- Updated Readme (explain OpenCV-CUDA support)

### Others
- Validation server:  Merged what remains
- Replaced sincos to sin and cos
- Handling of pointers in MPI programs
- "nullptr" used to denote the null pointer not "NULL"
- Check if nvidiaDriverVer is None

## Release 3.22.04 - Gaia

### Installation and user guide
- Updated readme
  - Updated hdf5 info troubleshoting
  - Updated Standalone installation 
  - Updated Scons installation 
- xmipp get_models: fixing the run and download path
- Updating xmipp links for Scipion on installation
- Removed fatal message in installation
- Reported error if happen on installation - runjob
- Ensuring that target directory for the libraries exists


### Protocols scipion-em-xmipp
- **protocol_core_analysis**: New protocol
- **protocol_compare_angles**: Bug fix in compare angles under some conditions
- **protocol_center_particles**: protocol simplified (removed setofCoordinates as output)
- **protocol_CTF_consensus**: concurrency error fixed
- **protocol_convert_pdb**: remove size if deactivated
- **protocol_resolution_deepres**: binary masked not stored in Extra folder and avoiding memory problems on GPUs
- **protocol_add_noise**: fixes
- **protocol_compare_reprojections**: improve computation of residuals + tests + fix + formatting
- **protocol_screen_deepConsensus**: multiple fixes in batch processing, trainging and streaming mode
- **protocol_shift_particles**: apply transform is now optional
### Others
- New XMIPP logo
- subtract_projection: adding new flag + fix
- Add intersection size metadata (bindings/python)
- Fixed unitialized unique pointers (bindings/python)
- Bug fixing: Resolution directional and anisotropic filtering fixing the test
- Removed SonarCloud issues
  - Replaced defines with constexpr
  - Removing Unused funtion parameters
  - Division by zero
  - Memory management
  - Removed field shadowing
  - Destructors should not throw exceptions

## Release 3.22.01 - Eris


- Updating to C++17
- Support newer versions of CUDA and gcc
- Zernike programs compatible with Cuda 8.x
- Fixed Sonar Cloud issues and bugs
- Matlab compilation Fixed
- Fixed importing pwem.metadata
- nma_alignment: Fixed arguments for the xmipp_angular_projection_matching invocation
- Fixed test  fails: ResolutionSsnr, ReconstructArtMpi, ReconstructArt, MlfRefine3dMpi, MlfRefine3d, MlRefine3dMpi, MlRefine3d, xmipp_test_pocs_main & volume_subtraction
- xmipp_micrograph_automatic_picking: Fixed tests, avoid possible memory corruption
- resolution_pdb_bfactor: bug fixed -  error with multiple chains
- FlexAlign: Fixed crash when binning > 1
- Bug fixed and allowed controlling high sampling rate
- Volume consensus: Fixed number of levels in the wavelet transform
- Compilation: Fixed compilation of starpu programs
- xmipp_transform_dimred: Fixed output metadata in append mode, adding MDL_DIMRED label
- Config file generation: Fixed config version detection outside of the git repo,  refactored check_CUDA and managed gcc compiler if it is installed out of /usr/bin/, check and exit if xmipp.conf does not exist
- Compilation: Fixed detection of the last commit changed the config script
- Resolution_fso: Bingham test implemented
- Opencv not detected. Added include to user/include/opencv4 folder on config file
- Compilation: asking whether to continue with compilation even though the config file is outdated
- XMIPP install: Linked libsvm to scipion
- Installation: Referenced 'global' xmipp.conf instead of using local copy of it
- Multiple MPI programs: replaced CREATE_MPI_METADATA_PROGRAM macro by templated class
- python_constants: add defocus labels
- Metadata: added new nmaEigenval label
- Python binding: added new function -  correlationAfterAlignment, MDL_RESOLUTION_ANISOTROPY, MDL_RESOLUTION_ANISOTROPY
- Matlab binding dependencies: set XMIPP as a hard dependency
- Projections subtraction: new program
- FFTwT: added mutex for plan handling
- Multiple programs: Added a common implementation of the rerun
- Phantom_create: update info link
- Multiple programs: Added a common implementation of the rerun
- Transform Geometry: save new shifted coordinates in option "shift to" + enterOfMass to python binding 
- Readme info: add virtual machine info
- Removal of the SVM from inside the XMIPP repository and downloading it as an external dependence
- Solved a configuration problem with CUDA
- ml_tomo: Using .mrc instead of .vol ; volume_align: Addded wrapping during alignment






## Release 3.21.06 - Caerus

- CUDA-11 support
- New protocol: Deep align
- ChimeraX support
- Improvements of streaming process
- Several performance optimizations
- Build time optimization
- Multiple bug fixes
- Improved documentation


## Release 3.20.07 - Boreas

- Fast CTF estimation
- CTF includes phase shifts now
- Selection of alpha helices or beta sheets from a PDB (xmipp_pdb_select)
- Centering a PDB (xmipp_pdb_center)
- New Protocol: MicrographCleaner is a new algorithm that removes coordinates picked from carbon edges, aggregations, ice crystals and other contaminations
- New functionality: The protocol compare reprojections can now compute the residuals after alignment
- New protocol: Split frames divide input movies into odd and even movies so that they can be processed independently
- New protocol: Continuous heterogeneity analysis using spherical harmonics (not ready to be used)
- Bug fixing when some micrograph has no coordinates in the consensus-picking.
- New functionalities: Different architectures and training modes
- Normal Mode Analysis protocols have been moved to the plugin ContinuousFlex
- Fixing MPI version of the Fourier Reconstruction
- New protocol: local CTF integration and consensus protocol for local ctf (also the viewers)
- Local CTF analysis tools: Not yet ready for general public
- New functionallity: Introducing the posibility of automatic estimation of the gain orientation.
- Bugs fixings regarding stability on streaming processing
- Support of heterogeneous movie sets
- New protocol: Clustering of subtomogram coordinates into connected components that can be processed independently
- New Protocol: Removing duplicated coordinates
- New protocol: Subtomograms can be projected in several ways to 2D images so that 2D clustering tools can be used
- New protocol: Regions of Interest can be defined in tomograms (e.g., membranes)
- Bug fixing in mask3d protocol
- Bug fix: in helical search symmetry protocol
- Enhanced precision of the FlexAlign program
- Now, deepLearningToolkit is under its own conda environment
- Multiple protocols accelerated using GPU
- New functionality: Xmipp CTF estimation can now take a previous defocus and do not change it
- New functionallity: CTF-consensus is able to take the primary main values or an average of the two.
- New functionallity: CTF-consensus is able to append metadata from the secondary input
- New functionality: Xmipp Highres can now work with non-phase flipped images
- New functionality: Xmipp Preprocess particles can now phase flip the images
- New protocol: Tool to evaluate the quality of a map-model fitting
- Allowing multi-GPU processing using FlexAlign
- Improvement in monores and localdeblur
- Randomize phases also available for images
- Change the plugin to the new Scipion structure
- Migrating the code to python3
