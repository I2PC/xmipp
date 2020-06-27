XMIPP_CONDA_ENVS = {
  "xmipp_DLTK_v0.3": {
    "pythonVersion": "3",
    "dependencies": ["pandas=0.23", "scikit-image=0.14", "opencv=3.4",
                     "tensorflow%(gpuTag)s=1.10", "keras=2.2",
                     "scikit-learn=0.22"],
    "channels": ["anaconda"],
    "pipPackages": [],
    "defaultInstallOptions": {"gpuTag": ""},
    "xmippEnviron": True
  },

  "xmipp_MicCleaner": {
    "pythonVersion": "3.6",
    "dependencies": ["micrograph-cleaner-em=0.35"],
    "channels": ["rsanchez1369", "anaconda", "conda-forge"],
    "pipPackages": [],
    "defaultInstallOptions": {},
    "xmippEnviron": False
  }

}
