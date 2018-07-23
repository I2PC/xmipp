#ifndef CUFFTADVISOR_UTILS_H_
#define CUFFTADVISOR_UTILS_H_

#include <cstring>
#include <sstream>
#include <vector>

namespace cuFFTAdvisor {
#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)
#define SEP "|"

template <typename T>
static inline std::string numToString(T num) {
  std::stringstream ss;
  ss << num;
  return ss.str();
}

// define utilities
static inline float toMB(size_t bytes) { return bytes / 1048576.f; }

template <typename T>
static inline void deleteEach(std::vector<T *> &v) {
  typename std::vector<T *>::iterator it;
  for (it = v.begin(); it != v.end(); ++it) {
    delete *it;
  }
}

template <typename T>
struct Triplet {
  Triplet(T f, T s, T r) : fst(f), snd(s), rd(r) {}
  T fst;
  T snd;
  T rd;
};

namespace Tristate {
enum Tristate { FALSE = 0, TRUE = 1, BOTH = 2 };

static inline bool is(Tristate tr) { return (TRUE == tr) || (BOTH == tr); }
static inline bool isNot(Tristate tr) { return (FALSE == tr) || (BOTH == tr); }

static inline const char *toString(Tristate t) {
  if (t == FALSE) return "false";
  if (t == TRUE) return "true";
  return "both";
}

}  // namespace Tristate

namespace CudaVersion {  // FIXME implement auto detection
enum CudaVersion { V_8, V_9 };
}  // namespace CudaVersion

static inline const char *toString(cuFFTAdvisor::Tristate::Tristate t) {
  switch (t) {
    case Tristate::TRUE:
      return "true";
    case Tristate::FALSE:
      return "false";
    case Tristate::BOTH:
      return "both";
    default:
      return "Tristate undefined";
  }
}

static inline bool safeEquals(const char *l, const char *r) {
  if ((NULL == l) || (NULL == r))
    return l == r;
  else
    return (0 == std::strcmp(l, r));
}

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_UTILS_H_
