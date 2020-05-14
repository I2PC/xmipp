XMIPP_CONDA_ENVS = {
  "xmipp_DLTK_v0.3": {
    "pythonVersion": "3",
    "dependencies": ["pandas=0.23", "scikit-image=0.14", "opencv=3.4",
                     "tensorflow%(gpuTag)s==1.10", "keras=2.2",
                     "scikit-learn==0.22"],
    "channels": ["anaconda"],
    "pipPackages": [],
    "defaultInstallOptions": {"gpuTag": ""},
    "xmippEnviron": True
  },

  "xmipp_MicCleaner": {
    "pythonVersion": "3.6",
    "dependencies": ["numpy=1.16", "micrograph-cleaner-em", "keras=2.2"],
    "channels": ["rsanchez1369", "anaconda", "conda-forge"],
    "pipPackages": [],
    "defaultInstallOptions": {},
    "xmippEnviron": False
  },

  "xmipp_deepVolPostPro": {
    "pythonVersion": "3.6",
    "dependencies": ["tensorflow-gpu=1.14", "keras=2.2.4", "numba=0.45.1", "pandas=0.25", "scikit-image=0.15", "scikit-learn=0.21", "scipy=1.3", "tqdm=4"],
    "channels": ["rsanchez1369", "anaconda", "conda-forge"],
    "pipPackages": ['alt-model-checkpoint==1.13.0', 'git+https://www.github.com/keras-team/keras-contrib.git', 'mrcfile==1.1.2', 'keras-radam==0.12.0'],
    "defaultInstallOptions": {},
    "xmippEnviron": False
}


}
