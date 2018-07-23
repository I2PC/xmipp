#ifndef CUFFTADVISOR_INPUTPARSER_H_
#define CUFFTADVISOR_INPUTPARSER_H_

#include <climits>
#include <cstdio>
#include <cstdlib>
#include "utils.h"

namespace cuFFTAdvisor {

class InputParser {
 public:
  InputParser(int argc, char **argv);
  ~InputParser();
  bool reportUnparsed(FILE *stream);

  int x, y, z, n;
  int device;
  int maxSignalInc;
  int maxMemMB;
  bool allowTransposition;
  bool squareOnly;
  Tristate::Tristate isBatched;
  Tristate::Tristate isFloat;
  Tristate::Tristate isForward;
  Tristate::Tristate isInPlace;
  Tristate::Tristate isReal;

 private:
  InputParser(InputParser &other); // prohibit copy

  void parseDims();
  void parseDevice();
  int parseMaxSignalInc();
  int parseMaxMemMB();
  bool parseAllowTransposition();
  bool parseSquareOnly();
  Tristate::Tristate parseIsReal();
  Tristate::Tristate parseIsFloat();
  Tristate::Tristate parseIsForward();
  Tristate::Tristate parseIsBatched();
  Tristate::Tristate parseIsInPlace();

  int argc;
  char **argv;
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_INPUTPARSER_H_
