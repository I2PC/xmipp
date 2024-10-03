import os

_REQUIREMENT_PATH = os.path.join(os.path.dirname(__file__), 'envs_DLTK')

XMIPP_CONDA_ENVS = {
  "xmipp_DLTK_v0.3": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_DLTK_v0.3.yml'),
    "xmippEnviron": True
  },

  "xmipp_MicCleaner": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_MicCleaner.yml'),
    "xmippEnviron": False
  },

  "xmipp_deepEMhancer": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_deepEMhancer.yml'),
    "xmippEnviron": False
  },
  
  "xmipp_pyTorch": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_pyTorch.yml'),
    "xmippEnviron": True
  },

  "xmipp_DLTK_v1.0": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_DLTK_v1.0.yml'),
    "xmippEnviron": True
  },
  
  "xmipp_graph": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_graph.yml'),
    "xmippEnviron": True
  },

  "xmipp_cl2dClustering": {
    "requirements": os.path.join(_REQUIREMENT_PATH, 'xmipp_cl2d_clustering.yml'),
    "xmippEnviron": True
  },
}
