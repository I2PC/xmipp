#include "inputParser.h"

namespace cuFFTAdvisor {

InputParser::InputParser(int argc, char **argv) {
  x = y = z = n = device = INT_MAX;

  this->argv = new char *[argc];
  this->argc = argc;
  for (int i = 0; i < argc; i++) {
    this->argv[i] = argv[i];
  }

  parseDims();
  parseDevice();
  isReal = parseIsReal();
  isFloat = parseIsFloat();
  isForward = parseIsForward();
  isBatched = parseIsBatched();
  isInPlace = parseIsInPlace();
  maxSignalInc = parseMaxSignalInc();
  maxMemMB = parseMaxMemMB();
  allowTransposition = parseAllowTransposition();
  squareOnly = parseSquareOnly();
}

InputParser::~InputParser() {
  delete[] argv;
  argv = NULL;
}

int InputParser::parseMaxSignalInc() {
  for (int i = 0; i < (argc - 1); i++) {
    if (safeEquals(argv[i], "--maxSignalInc")) {
      if (NULL == argv[i + 1]) {
        return -1;
      }
      int perc = atoi(argv[i + 1]);
      argv[i] = argv[i + 1] = NULL;
      return perc;
    }
  }
  return INT_MAX;
}

int InputParser::parseMaxMemMB() {
  for (int i = 0; i < (argc - 1); i++) {
    if (safeEquals(argv[i], "--maxMem")) {
      if (NULL == argv[i + 1]) {
        return -1;
      }
      int mem = atoi(argv[i + 1]);
      argv[i] = argv[i + 1] = NULL;
      return mem;
    }
  }
  return INT_MAX;
}

bool InputParser::reportUnparsed(FILE *stream) {
  bool error = false;
  if (NULL != stream) {
    for (int i = 0; i < argc; i++) {
      if (NULL != argv[i]) {
        error = true;
        fprintf(stream, "unrecognized '%s'\n", argv[i]);
      }
    }
  }
  return error;
}

void InputParser::parseDims() {
  for (int i = 0; i < (argc - 1); i++) {
    if (safeEquals(argv[i], "-x")) {
      x = atoi(argv[i + 1]);
      argv[i] = argv[i + 1] = NULL;
    }
    if (safeEquals(argv[i], "-y")) {
      y = atoi(argv[i + 1]);
      argv[i] = argv[i + 1] = NULL;
    }
    if (safeEquals(argv[i], "-z")) {
      z = atoi(argv[i + 1]);
      argv[i] = argv[i + 1] = NULL;
    }
    if (safeEquals(argv[i], "-n")) {
      n = atoi(argv[i + 1]);
      argv[i] = argv[i + 1] = NULL;
    }
  }
  if (INT_MAX == y) {
    y = 1;  // user didn't put Y dim -> 1D transform
  }
  if (INT_MAX == z) {
    z = 1;  // user didn't put Z dim -> 2D transform
  }
  if (INT_MAX == n) {
    n = 1;  // user didn't put N dim -> no batch
  }
}

Tristate::Tristate InputParser::parseIsReal() {
  for (int i = 0; i < argc; i++) {
    if (safeEquals(argv[i], "--realOnly")) {
      argv[i] = NULL;
      return Tristate::TRUE;
    }
    if (safeEquals(argv[i], "--complexOnly")) {
      argv[i] = NULL;
      return Tristate::FALSE;
    }
  }
  return Tristate::BOTH;
}

Tristate::Tristate InputParser::parseIsFloat() {
  for (int i = 0; i < argc; i++) {
    if (safeEquals(argv[i], "--floatOnly")) {
      argv[i] = NULL;
      return Tristate::TRUE;
    }
    if (safeEquals(argv[i], "--doubleOnly")) {
      argv[i] = NULL;
      return Tristate::FALSE;
    }
  }
  return Tristate::BOTH;
}

Tristate::Tristate InputParser::parseIsForward() {
  for (int i = 0; i < argc; i++) {
    if (safeEquals(argv[i], "--forwardOnly")) {
      argv[i] = NULL;
      return Tristate::TRUE;
    }
    if (safeEquals(argv[i], "--inverseOnly")) {
      argv[i] = NULL;
      return Tristate::FALSE;
    }
  }
  return Tristate::BOTH;
}

Tristate::Tristate InputParser::parseIsInPlace() {
  for (int i = 0; i < argc; i++) {
    if (safeEquals(argv[i], "--inPlaceOnly")) {
      argv[i] = NULL;
      return Tristate::TRUE;
    }
    if (safeEquals(argv[i], "--outOfPlaceOnly")) {
      argv[i] = NULL;
      return Tristate::FALSE;
    }
  }
  return Tristate::BOTH;
}

Tristate::Tristate InputParser::parseIsBatched() {
  for (int i = 0; i < argc; i++) {
    if (safeEquals(argv[i], "--batchOnly")) {
      argv[i] = NULL;
      return Tristate::TRUE;
    }
    if (safeEquals(argv[i], "--noBatch")) {
      argv[i] = NULL;
      return Tristate::FALSE;
    }
  }
  return Tristate::BOTH;
}

void InputParser::parseDevice() {
  for (int i = 0; i < (argc - 1); i++) {
    if (safeEquals(argv[i], "-device")) {
      device = atoi(argv[i + 1]);
      argv[i] = argv[i + 1] = NULL;
      return;
    }
  }

  if (INT_MAX == device) {
    device = 0;  // user didn't put anything, use default one
  }
}

bool InputParser::parseAllowTransposition() {
  for (int i = 0; i < argc; i++) {
    if (safeEquals(argv[i], "--allowTransposition")) {
      argv[i] = NULL;
      return true;
    }
  }
  return false;
}

bool InputParser::parseSquareOnly() {
  for (int i = 0; i < argc; i++) {
    if (safeEquals(argv[i], "--squareOnly")) {
      argv[i] = NULL;
      return true;
    }
  }
  return false;
}

}  // namespace cuFFTAdvisor
