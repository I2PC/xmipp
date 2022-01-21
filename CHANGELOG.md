## Release 3.22.01 -


- Updating to C++17
- Support newer versions of CUDA and gcc
- Zernike programs compatible with Cuda 8.x
- Fixing Sonar Cloud issues and bugs
- Matlab compilation fix
- Resolution_fso: Bingham test implemented
- Fixing importing pwem.metadata
- nma_alignment: fixing arguments for the xmipp_angular_projection_matching invocation
- Fixed tests fails: ResolutionSsnr, ReconstructArtMpi, ReconstructArt, MlfRefine3dMpi, MlfRefine3d, MlRefine3dMpi, MlRefine3d, xmipp_test_pocs_main & volume_subtraction
- xmipp_micrograph_automatic_picking: fixing tests, avoiding possible memory corruption
- resolution_pdb_bfactor: bug fixing -  error with multiple chains
- FlexAlign: fixing crash when binning > 1
- Bug fix and allowed controlling high sampling rate
- Volume consensus: fixing number of levels in the wavelet transform
- Compilation: fixing compilation of starpu programs
- xmipp_transform_dimred: fixing output metadata in append mode, adding MDL_DIMRED label
- Config file generation: fixing config version detection outside of the git repo,  refactoring check_CUDA and managing gcc compiler if it is installed out of /usr/bin/, check and exit if xmipp.conf does not exist
- Compilation: Fixing detection of the last commit changing the config script
- Opencv not detected. Added include to user/include/opencv4 folder on config file
- Compilation: asking whether to continue with compilation even though the config file is outdated
- xmipp install: Linking libsvm to scipion
- Installation: Referencing 'global' xmipp.conf instead of using local copy of it
Multiple MPI programs: replacing CREATE_MPI_METADATA_PROGRAM macro by templated class
- python_constants: add defocus labels
- Metadata: adding new nmaEigenval label
- Python binding: adding new function -  correlationAfterAlignment, MDL_RESOLUTION_ANISOTROPY, MDL_RESOLUTION_ANISOTROPY
- Matlab binding dependencies, setting Xmipp as a hard dependency
- Projections subtraction: new program
- FFTwT: adding mutex for plan handling
- Multiple programs: Adding a common implementation of the rerun
- Ef subtract projection
- Phantom_create: update info link
- Multiple programs: Adding a common implementation of the rerun
- Transform Geometry: save new shifted coordinates in option "shift to" + enterOfMass to python binding
- config file: 
- Readme info: add virtual machine info
- Removal of the SVM from inside the Xmipp repository and downloading it as an external dependence
- Solving a configuration problem with CUDA
- ml_tomo: Using .mrc instead of .vol ; volume_align: Addding wrapping during alignment






## Release 3.21.06 -

- CUDA-11 support
- New protocol: Deep align
- ChimeraX support
- Improvements of streaming process
- Several performance optimizations
- Build time optimization
- Multiple bug fixes
- Improved documentation


## Release 3.20.07 -

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
