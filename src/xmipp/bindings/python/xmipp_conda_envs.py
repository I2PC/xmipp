XMIPP_CONDA_ENVS = {
  "xmipp_DLTK_v0.3": {
    "pythonVersion": "3.9",
    "dependencies": ["pandas=1.3.5",
                     "scikit-image=1.19.3",
                     "opencv=4.6",
                     "tensorflow%(gpuTag)s=1.15",
                     "keras=2.3",
                     "scikit-learn=0.22"],
    "channels": ["anaconda"],
    "pipPackages": {},
    "defaultInstallOptions": {"gpuTag": ""},
    "xmippEnviron": True
  },

  "xmipp_MicCleaner": {
    "pythonVersion": "3.6",
    "dependencies": ["micrograph-cleaner-em=0.35"],
    "channels": ["rsanchez1369", "anaconda", "conda-forge"],
    "pipPackages": {},
    "defaultInstallOptions": {},
    "xmippEnviron": False
  },

  "xmipp_deepEMhancer": {
    "pythonVersion": "3.6",
    "dependencies": ["deepemhancer=0.12", "numba=0.45"],
    "channels": ["rsanchez1369", "anaconda", "conda-forge"],
    "pipPackages": {},
    "defaultInstallOptions": {},
    "xmippEnviron": False
  },
  "xmipp_deepHand": {
    "pythonVersion": "3.8",
    "dependencies": ["pytorch=1.6"],
    "channels": ["anaconda", "conda-forge"],
    "pipPackages": {},
    "defaultInstallOptions": {},
    "xmippEnviron": True
}


}
