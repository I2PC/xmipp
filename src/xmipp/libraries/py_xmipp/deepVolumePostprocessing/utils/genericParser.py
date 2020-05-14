import argparse

from ..configManager import ConfigManager

from ..config import VOLUMES_PATH


def parseProcessingType(description, additionalArgs=[], skypFileOfIds=False):
  '''

  :param additionalArgs: [ (*args, **kwards ), ]
  :return:
  '''

  parser = argparse.ArgumentParser(description)
  if not skypFileOfIds:
    parser.add_argument('-f', '--fileOfIds', type=str, nargs=None, required=False, default=None,
                        help='read emdbIds from file instead of using all in path:\n'+VOLUMES_PATH)
  parser.add_argument('--locscale', action='store_true', required=False,
                      help='work using locscale. Excludes --tightMask, --pdb, --mask and --improveRes')

  parser.add_argument('--tightMask', action='store_true', required=False,
                      help='work using tightMask. Excludes --locscale, --pdb, --mask and --improveRes')

  parser.add_argument('--mask', action='store_true', required=False,
                      help='work using wide mask. Excludes --locscale, --pdb, --tightMask and --improveRes')

  parser.add_argument('--pdb', action='store_true', required=False,
                      help='work using pdb. Excludes --locscale, --tightMask, --improveRes and --mask')

  parser.add_argument('--improveRes', action='store_true', required=False,
                      help='work using improve resolution. Excludes --tightMask, --pdb, --mask and --locscale')

  for kwarg in additionalArgs:
    parser.add_argument(*kwarg[:-1], **kwarg[-1])

  args= parser.parse_args()
  unicity_error= 'Only one of the following options --tightMask, --mask, --locscale, --pdb or --improveRes must be provided'

  processingType=None
  if args.locscale:
    processingType= "locscale"

  if args.improveRes:
    if processingType is not None:
      parser.error(unicity_error)
    processingType= "improveRes"

  if args.tightMask:
    if processingType is not None:
      parser.error(unicity_error)
    processingType= "tightMask"

  if args.pdb:
    if processingType is not None:
      parser.error(unicity_error)
    processingType= "pdb"

  if args.mask:
    if processingType is not None:
      parser.error(unicity_error)
    processingType= "mask"

  if processingType is None:
    parser.error(unicity_error)

  ConfigManager.setProcessingType(processingType)
  return processingType, args
