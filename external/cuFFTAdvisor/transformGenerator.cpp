#include "transformGenerator.h"

namespace cuFFTAdvisor {

void TransformGenerator::generate(int device, int x, int y, int z, int n,
                                  bool isBatched, bool isFloat, bool isForward,
                                  bool isInPlace, Tristate::Tristate isReal,
                                  std::vector<Transform const *> &result) {
  if (Tristate::FALSE != isReal) {
    result.push_back(new Transform(device, x, y, z, n, isBatched, isFloat,
                                   isForward, isInPlace, true));
  }
  if (Tristate::TRUE != isReal) {
    result.push_back(new Transform(device, x, y, z, n, isBatched, isFloat,
                                   isForward, isInPlace, false));
  }
}

void TransformGenerator::generate(int device, int x, int y, int z, int n,
                                  bool isBatched, bool isFloat, bool isForward,
                                  Tristate::Tristate isInPlace,
                                  Tristate::Tristate isReal,
                                  std::vector<Transform const *> &result) {
  if (Tristate::FALSE != isInPlace) {
    generate(device, x, y, z, n, isBatched, isFloat, isForward, true, isReal,
             result);
  }
  if (Tristate::TRUE != isInPlace) {
    generate(device, x, y, z, n, isBatched, isFloat, isForward, false, isReal,
             result);
  }
}

void TransformGenerator::generate(int device, int x, int y, int z, int n,
                                  bool isBatched, bool isFloat,
                                  Tristate::Tristate isForward,
                                  Tristate::Tristate isInPlace,
                                  Tristate::Tristate isReal,
                                  std::vector<Transform const *> &result) {
  if (Tristate::FALSE != isForward) {
    generate(device, x, y, z, n, isBatched, isFloat, true, isInPlace, isReal,
             result);
  }
  if (Tristate::TRUE != isForward) {
    generate(device, x, y, z, n, isBatched, isFloat, false, isInPlace, isReal,
             result);
  }
}

void TransformGenerator::generate(int device, int x, int y, int z, int n,
                                  bool isBatched, Tristate::Tristate isFloat,
                                  Tristate::Tristate isForward,
                                  Tristate::Tristate isInPlace,
                                  Tristate::Tristate isReal,
                                  std::vector<Transform const *> &result) {
  if (Tristate::FALSE != isFloat) {
    generate(device, x, y, z, n, isBatched, true, isForward, isInPlace, isReal,
             result);
  }
  if (Tristate::TRUE != isFloat) {
    generate(device, x, y, z, n, isBatched, false, isForward, isInPlace, isReal,
             result);
  }
}

void TransformGenerator::generate(int device, int x, int y, int z, int n,
                                  Tristate::Tristate isBatched,
                                  Tristate::Tristate isFloat,
                                  Tristate::Tristate isForward,
                                  Tristate::Tristate isInPlace,
                                  Tristate::Tristate isReal,
                                  std::vector<Transform const *> &result) {
  if (Tristate::FALSE != isBatched) {
    generate(device, x, y, z, n, true, isFloat, isForward, isInPlace, isReal,
             result);
  }
  if (Tristate::TRUE != isBatched) {
    generate(device, x, y, z, n, false, isFloat, isForward, isInPlace, isReal,
             result);
  }
}

void TransformGenerator::transpose(GeneralTransform &tr,
                                   std::vector<GeneralTransform> &result) {
  std::set<Triplet<int>, TransposeComp> candidates;
  candidates.insert(Triplet<int>(tr.X, tr.Y, tr.Z));
  if (tr.Y != 1) {
    candidates.insert(Triplet<int>(tr.Y, tr.X, tr.Z));
  }
  if (tr.Z != 1) {
    candidates.insert(Triplet<int>(tr.X, tr.Z, tr.Y));
    candidates.insert(Triplet<int>(tr.Y, tr.Z, tr.X));
    candidates.insert(Triplet<int>(tr.Z, tr.X, tr.Y));
    candidates.insert(Triplet<int>(tr.Z, tr.Y, tr.X));
  }

  std::set<Triplet<int> >::iterator it;
  for (it = candidates.begin(); it != candidates.end(); ++it) {
    result.push_back(GeneralTransform(it->fst, it->snd, it->rd, tr));
  }
}

}  // namespace cuFFTAdvisor
