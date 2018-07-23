#include "validator.h"

namespace cuFFTAdvisor {

void Validator::validate(int x, int y, int z, int n, int device,
                         int maxSignalInc, int maxMemMB,
                         bool allowTrans, bool squareOnly) {
  validate(x, y, z, n, device);

  if (maxSignalInc <= 0) {
    throw std::logic_error(
        "Max signal (perc) increase must be positive. Wrong input or int "
        "overflow\n");
  }

  if (maxMemMB <= 0) {
    throw std::logic_error(
        "Max memory must be positive. Wrong input or int overflow\n");
  }

  if (maxMemMB > std::ceil(toMB(getTotalMemory(device)))) {
    throw std::logic_error("Selected device does not have that much memory.\n");
  }

  if (allowTrans && squareOnly) {
    throw std::logic_error("Incompatible parameters. See help for detailed info.\n");
  }
}

void Validator::validate(int device) {
  int devices = getDeviceCount();  // This function call returns 0 if
                                   // there are no CUDA capable devices.
  if (0 == devices) {
    throw std::logic_error(
        "There are no available device(s) that support CUDA\n");
  }
  if ((device < 0) || (device >= devices)) {
    throw std::logic_error(
        "No such CUDA device available. Wrong input or int overflow\n");
  }
}

void Validator::validate(int x, int y, int z, int n, int device) {
  if ((x <= 0) || (INT_MAX == x)) {
    throw std::logic_error(
        "X dim must be positive and < INT_MAX. Wrong input or int overflow.\n");
  }
  if (y <= 0) {
    throw std::logic_error(
        "Y dim must be positive. Wrong input or int overflow.\n");
  }
  if (z <= 0) {
    throw std::logic_error(
        "Z dim must be positive. Wrong input or int overflow.\n");
  }
  if (n <= 0) {
    throw std::logic_error(
        "N dim must be positive. Wrong input or int overflow.\n");
  }
  if (device < 0) {
    throw std::logic_error("Device identifier must be positive. Wrong input.");
  }

  validate(device);
}

}  // namespace cuFFTAdvisor
