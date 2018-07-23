#ifndef CUFFTADVISOR_TRANSFORMGENERATOR_H_
#define CUFFTADVISOR_TRANSFORMGENERATOR_H_

#include <set>
#include "generalTransform.h"
#include "transform.h"
#include "utils.h"

namespace cuFFTAdvisor {

class TransformGenerator {
 public:
  static void generate(int device, int x, int y, int z, int n,
                       Tristate::Tristate isBatched, Tristate::Tristate isFloat,
                       Tristate::Tristate isForward,
                       Tristate::Tristate isInPlace, Tristate::Tristate isReal,
                       std::vector<Transform const *> &result);

  static void generate(int device, int x, int y, int z, int n, bool isBatched,
                       bool isFloat, bool isForward, bool isInPlace,
                       Tristate::Tristate isReal,
                       std::vector<Transform const *> &result);

  static void generate(int device, int x, int y, int z, int n, bool isBatched,
                       bool isFloat, bool isForward,
                       Tristate::Tristate isInPlace, Tristate::Tristate isReal,
                       std::vector<Transform const *> &result);

  static void generate(int device, int x, int y, int z, int n, bool isBatched,
                       bool isFloat, Tristate::Tristate isForward,
                       Tristate::Tristate isInPlace, Tristate::Tristate isReal,
                       std::vector<Transform const *> &result);

  static void generate(int device, int x, int y, int z, int n, bool isBatched,
                       Tristate::Tristate isFloat, Tristate::Tristate isForward,
                       Tristate::Tristate isInPlace, Tristate::Tristate isReal,
                       std::vector<Transform const *> &result);

  static void transpose(GeneralTransform &tr,
                        std::vector<GeneralTransform> &result);

 private:
  struct TransposeComp {
    bool operator()(const Triplet<int> &l,
                    const Triplet<int> &r) {  // FIXME move
      if (l.fst != r.fst) return l.fst < r.fst;
      if (l.snd != r.snd) return l.snd < r.snd;
      return l.rd < r.rd;
    }
  };
};

}  // namespace cuFFTAdvisor

#endif  // CUFFTADVISOR_TRANSFORMGENERATOR_H_
