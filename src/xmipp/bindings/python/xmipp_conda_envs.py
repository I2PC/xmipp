XMIPP_CONDA_ENVS = {
  "xmipp_DLTK_v0.3": {
    "pythonVersion": "3",
    "dependencies": ["pandas=0.23", "scikit-image=0.14", "opencv=3.4",
                     "tensorflow%(gpuTag)s=1.15", "keras=2.2",
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
  },

  "xmipp_deepEMhancer": {
    "pythonVersion": "3.6",
    "dependencies": ["deepemhancer=0.12", "numba=0.45"],
    "channels": ["rsanchez1369", "anaconda", "conda-forge"],
    "pipPackages": [],
    "defaultInstallOptions": {},
    "xmippEnviron": False
  },
  "xmipp_deepHand": {
    "pythonVersion": "3.8",
    "dependencies": ["pytorch=1.6"],
    "channels": ["anaconda", "conda-forge"],
    "pipPackages": [],
    "defaultInstallOptions": {},
    "xmippEnviron": True
  },
  "xmipp_torch": {
    "pythonVersion": "3.9",
    "dependencies": ["pytorch", "pytorch-cuda=11.6", "torchvision", # FIXME pytorch-cuda only when gpu is enabled
                     "pandas", "faiss%(gpuTag)s", "numpy" ],
    "channels": ["pytorch", "nvidia", "anaconda", "conda-forge"], # FIXME nvidia only when GPU is enabled
    "pipPackages": [],#["mrcfile", "starfile"],
    "defaultInstallOptions": {"gpuTag": "-cpu"},
    "xmippEnviron": True
}

}
