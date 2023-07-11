   ## Release 3.23.07 - voting
   ### Xmipp Programs 
   - New programs: angular_resolution_alignment, image_peak_high_contrast(for detecting high contrast regions in tomographic reconstruction), misaligment_detection (to detect misalignment in tomographic reconstructions from high-contrast regions)
   - Deprecated programs: classify_kmeans_2D, rotational_spectra, particle_boxsize. (For more details visit [this]([url](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols)) 
   - xmipp_angular_distance: new features
   - tomo_extract_particles: new features
   - subtract_projection: parallelization with mpi
   - angular_project_library: Removed deterministic behaviour (mpi)
   - volumen_subtraction: fixed bug
   - tomo_extract_subtomograms: allow downsampling of features
   - angular_resolution_alignment:  Detect misalignment with resolution
   - align_volume_and_particles: Fixed error



   ### Protocols scipion-em-xmipp
   - New protocols: Movie Dose analysis, consensus_classes (More efficient p-value calculation, 
Updated intersection merging process, generalized protocol for other set of classes)
   - Tilt analysis: Close correctly the output sets once finished
   -Deep micrograph cleaner: fix two bugs that occured during streaming implementation bug 
   - volume_adjust_sub: fix with :mrc
   - Picking consensus: define correctly the possibleOutputs bug 
   - Center particles: streaming bug when definining the outputs bug 
   - subtract_projection: change pad validation error for warning
   - Movie Gain: changed _stepsCheckSecs and fixed inputMovies calling
   - convert_pdb: dont allow set size if template volume
   - CTF_consensus: add 4 threads by default
   - particle_pick_automatic: Improved conditions
   - process: Better instantiation of Scipion subclasses
   - monores_viewer: fix histogram
   - create_mask3d: Addding a validate in 3dmask
   - consensus_local_ctf: save defocus in proper fields
   - create3dMaks: add :mrc to input filename
   - align_volume: Included the label in the volumes, change the label for each volume
   - volume_subtraction: bug fixed in filename
   - viewer_resolution_fs: fixing 0.1 threshold not found
   - consensus_local_ctf improvement: compute consensus for local defocus U and V separately
   - xmipp-movie-gain: np.asscalar discontinued in numpy 1.16
   - viewer_projmatch, viewer_metaprotocol_golden_highres: Fixing viewers, change removed ChimeraClientView to ChimeraView
   - convert_pdb: generates an mrc file
   - compare_reprojections: fix update subtract projection output
   - deep_micrograph_screen: Bug fix that prevents using small GPUs
   - consensus_local_ctf: add consensus angle
   - crop_resize: Add mask as input. Mask resize is now possible
   - align_volume: Do not wrap
   - consensus_classes:Fixed manual output generation
   - reconstruct_significant: new test
   - convert_pdb: to convert a set of pdbs to volumes







     
   -  Protocols deprecated: apply_deformation_zernike3d, classify_kmeans2d, kmeans_clustering, particle_boxSize, rotational_spectra, split_volume_hierarchical_cluster (For more details visit [this]([url](https://github.com/I2PC/xmipp/wiki/Deprecating-programs-and-protocols)) 

   -
   
   
   ### Installation and user guide
   - New clearer, more intuitive and informative installer. It also creates a file to facilitate user support.
   - Refactored the deep learning toolkit, more robust and new environment with updated tensorflow.
   - Updated requirement.
     
   ### More Xmipp 
   - Added half precission suport to numpy
   - Added the ability to read and write CIF files
   - Modular design of winner filter
   - Decoupling XmippTomo from XmippSPA
   - Fix Zernike equation
   - Removed all occurrences of non base-class default destructors
   - Improved MultidimArray performance
   - Added support for half precision floating point numbers in MRCs
   - Assign tiff to gain files
   - adding half maps labels  


   
   ## Release 3.23.03 - Kratos
   ### Xmipp Programs 
   
  - New programs: tomo_confidence_map, tomo_extract_particlestacks, tomo_extract_subtomograms, tomo_tiltseries_dose_filter, psd_estimatator
  - Deprecated programs (for more details visit the [wiki](https://github.com/I2PC/xmipp/wiki/Deprecating-programs)):
    angular_distribution_show, apropos
    ctf_correct_idr, ctf_create_ctfdat , ctf_show , idr_xray_tomo , image_common_lines , metadata_convert_to_spider , metadata_selfile_create , mlf_refine_3d, ml_refine_3d, ml_tomo , mrc_create_metadata , pdb_construct_dictionary, pdb_restore_with_dictionary , reconstruct_admn , reconstruct_art_pseudo , resolution_ibw , resolution_ssnr , score_micrograph , reconstruct_fourier_starpu , tomo_align_tilt_series, tomo_align_dual_tilt_series, tomo_align_refinement, tomo_align_refinement, tomo_extract_subvolume, tomo_project_main, tomo_remove_fluctuations , tomo_align_tilt_series,transform_range_adjust , validation_tilt_pairs , volume_pca , volume_validate_pca , work_test , 6f4d983 , evaulate_coordinates , extract_subset , image_separate_objects , volume_enhance_contrast , volume_reslice , xray_import , xray_project , xray_psf_create , xray_reconstruct_art , gpu_correlation, gpu_utils, classify_significant, deepAlign.

  - volume_from_pdb: fixing input pdb file being overwritten when '-centerPDB' flag was set
  - xmipp_phantom_movie:  adding support for fixed step shift & gain and dark image generation
  - CTF simulation allows astigmatism
  - xmipp_metadata_utility: Now join operations with an empty set will return a new empty set (previously no output file was generated).
  - xmipp_matrix_dimred: Program help improved. Exception is now thrown when the number of output dimensions is larger than the input dimensions
  - xmipp_angular_distance: Added itemId column to the output


   ### Protocols scipion-em-xmipp
  - New protocol status: beta, new, production and updated. Will appear in the left pannel of Scipion 
  - Protocol subtract_projection: user experience improvements, no final mask by default, apply ciruclar mask in adjustment image to avoid edge artifacts, validate same sampling rate with tolerance in third decimal
  - Protocol convert_pdb: Allowed to save centered PDB used for conversion. 
  - Protocol align_volume_and_particles: add alingment validation
  - Protocol FlexAlign: updating protocol to reflect changes in the executable, fixed test, removing unused protocol (Movie average)
  - Protocol align_volume_and_particles:Align volume and particles adapted to tomography and works in the absence of tomo plugin.
  - Protocol volume_consensus: validate same sampling rate with tolerance in third decimal
  - Protocols deprecated (for more details visit the [wiki](https://github.com/I2PC/xmipp/wiki/Deprecating-programs)): protocol_deep _align, reconstruct_heterogeneous, protocol_metaprotocol_create_output, protocol_metaprotocol_discrete_heterogeneity_scheduler
  
   ### Installation and user guide
   - Refactor and simplified Readme page.
   - Updating CUDA version compatibility
   - Updating gcc version availables
   - Fixed Matlab installation
   - Added missing array include to fix compilation error with g++12
   - Alert and not block compilation if gcc - CUDA are not compatible
   - Avoid compilation warnings
   - Required pyworkflow==3.0.31


   ### Others
   - Maintenance: Recovered python binding tests
   - Maintenance: fixing dangling pointer in xmipp_error
   - Maintenance: Cleaned includes in xmipp_image_base
   - PSD estimation: templating function, improving performance
   - Flag cleanDeprecate in the installation; clean all deprecated executables programs 
   - python binding: fixed bug when Numpy arrays created by slicing were badly interpretted
   - Removed "seed" library
   - Fixed memory pinning CUDA bug
   - Fixed compilation errors on CUDA 9



   ## Release 3.22.11 - Iris

  ### Xmipp Programs
  -  Speeding up iterations in some xmipp programs (xmipp_ctf_group, xmipp_image_histogram, xmipp_mpi_angular_class_average, xmipp_angular_distance, xmipp_angular_estimate_tilt_axis, xmipp_ctf_create_ctfdat, xmipp_resolution_ssnr)
  -  New Zernike3D programs
  -  angular_project_library: Reported some error if there are no images in the range
  -  angular_discrete_assign.cpp: Removed memory leak and uninitialized values
  -  angular_distance: Fixing condition to avoid iteration behind the end of the MD in cases when input data have different sizes. Optimized performance
  -  Pdb_reduce_pseudoatoms: Produced pdb is one-based indexed
  -  xmipp_micrograph_automatic_picking: Fixing memory leak
  -  subtract_projection: Fixed several bugs (improved results), added circular mask to avoid edge artifacts, added option to boost particles instead of subtract


  ### Protocols scipion-em-xmipp
  - Protocol_cl2d_align: The input can now be a set of averages or a set of 2D classes 
  - Protocol_local_ctf: Default value are now changed for maxDefocusChange
  - Protocol_apply_zernike3d: Now accepts either a Volume or SetOfVolumes and applies the coefficients in a loop in the deform step
  - Protocol_postProcessing_deepPostProcessing: Managed GPU memory to avoid errors
  - Protocol_resolution_deepres: Mandatory mask
  - Protocol center particles and Gl2d (all options): Fix streaming
  - Protocol_create_3d_mask: Allows volume Null=True
  - Protocol_reconstruct_fourier: Set pixel size
  - GL2D static: Bug fixing
  - Protocol_trigger_data: Bug fixing
  - Protocol_crop_resize: Set sampling rate of mrc files when cropping resizing volumes or particles
  - subtract_projection: New protocol for boosting particles. Add protocol to wizard XmippParticleMaskRadiusWizard as now the protocol uses it

  - **New tests:** deep_hand, pick_noise, screen_deep_learning, resolution_B_factor
  - Fixed TestHighres test
  
  ### Installation and user guide
  - Various bug fixing
  - More information about hdf5 library
  - Updating CUDA - GCC compatibility. Added CUDA 11.7 (not tested)
  - Updating Readme
  
  ### Others
  - Performance optimization (metadata binding)
  - Python binding: adding methods to directly set / get entire MD row
  - g++ >= 8 required
  - In viewers used pwutils 
  - The pdb data library now has all the right fields and should write the record type ("ATOM " or "HETATM") correctly at the beginning of the line and the atomType (element) and charge (if applicable) correctly at the end of the line.
  - Removal of an artifact of symmetrization related to the z pitch (symmetries.cpp)
  - Using the same identical Deprecated param from pyworkflow.


## Release 3.22.07 - Helios


### Scripts Xmipp
- **xmipp_image_operate**: taked into account non existing files
- **angular_continuous_assign2**: Bug fixed
- **volume_consensus**: Bug fixed
- **ctf.h and angular_continuous_assign_2**: Changes for local defocus estimation #578

### Protocols scipion-em-xmipp
- **rotate_volume**: New protocol
- **subtract_projection**: New implementation based on adjustment by regression instead of POCS and improved performance
- **local_ctf**: Add new sameDefocus option + formatting
- **compare_reprojections & protocol_align_volume**: Fast Fourier by default
- **crop_resize**: Allows input pointers
- **resolution_deepres**: Resize output to original size
- **denoise_particles**: Added setOfAverages as input option
- **process**: Change output from stk (spider) to mrcs (mrc)
- **trigger_data**: Bug fixed
- **screen_deeplearning**:  Added descriptive help" 
- **center_particles**: Added summary info
- **align_volume_and_particles**: Summary error fixed
- **cl2d**: Summary errors solved 

- **New tests:** test_protocol_reconstruct_fourier, test_protocols_local_defocus, test_protocols_local_defocus, TestXmippAlignVolumeAndParticles,  TestXmippRotateVolume
- **Improved tests:** test_protocols_deepVolPostprocessing, test_protocols_xmipp_3d, Test ProjSubtracion
- **Excluded tests:** test_protocols_zernike3d, test_protocols_metaprotocol_heterogeneity


### Installation and user guide
- Version info printed at the end of the installation
- Removed empty folder with cleanBin command
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
