#include "sizeOptimizer.h"

namespace cuFFTAdvisor {

const struct SizeOptimizer::Polynom SizeOptimizer::UNIT = {
    .value = 1, 0, 0, 0, 0, 0, 0};

SizeOptimizer::SizeOptimizer(CudaVersion::CudaVersion version,
                             GeneralTransform &tr, bool allowTrans)
    : version(version),
      log_2(1.0 / std::log(2)),
      log_3(1.0 / std::log(3)),
      log_5(1.0 / std::log(5)),
      log_7(1.0 / std::log(7)) {
  if (Tristate::BOTH == tr.isFloat) {
    // if user is not sure if he/she needs double, then he/she doesn't need it
    tr.isFloat = Tristate::TRUE;
  }

  if (allowTrans) {
    std::vector<GeneralTransform> transposed;
    TransformGenerator::transpose(tr, transposed);
    input.insert(input.end(), transposed.begin(), transposed.end());
  } else {
    input.push_back(tr);
  }

#ifdef DEBUG
  std::vector<GeneralTransform>::iterator it;
  for (it = input.begin(); it != input.end(); ++it) {
    it->print();
  }
#endif
}

std::vector<const Transform *> *SizeOptimizer::optimize(size_t nBest,
                                                        int maxPercIncrease,
                                                        int maxMemMB, bool squareOnly) {
  std::vector<GeneralTransform> preoptimized;
  size_t origSize = input.size();
  for (size_t i = 0; i < origSize; ++i) {
    std::vector<GeneralTransform> *tmp =
        optimizeXYZ(input.at(i), nBest, maxPercIncrease, squareOnly);
    preoptimized.insert(preoptimized.end(), tmp->begin(), tmp->end());
    delete tmp;
  }
  return optimizeN(&preoptimized, maxMemMB, nBest);
}

bool SizeOptimizer::sizeSort(const Transform *l, const Transform *r) {
  if (l->N != r->N) return l->N > r->N;  // prefer bigger batches
  size_t lDims = l->X * l->Y * l->Z;
  size_t rDims = r->X * r->Y * r->Z;
  if (lDims != rDims) return lDims < rDims;
  if (l->Z != r->Z) return l->Z < r->Z;
  if (l->Y != r->Y) return l->Y < r->Y;
  return l->X < r->X;
}

bool SizeOptimizer::perfSort(const Transform *l, const Transform *r) {
  if (l->isFloat && (!r->isFloat)) return true;
  if ((!l->isFloat) && r->isFloat) return false;
  // both float or double
  if (l->isReal && (!r->isReal)) return true;
  if ((!l->isReal) && r->isReal) return false;
  // both complex or real
  if (l->isInPlace && (!r->isInPlace)) return false;
  if ((!l->isInPlace) && r->isInPlace) return true;
  // both in-place or out-of-place
  if (l->isBatched && (!r->isBatched)) return true;
  if ((!l->isBatched) && r->isBatched) return false;
  // both batched or not batched
  return sizeSort(l, r);
}

std::vector<const Transform *> *SizeOptimizer::optimizeN(
    std::vector<GeneralTransform> *transforms, size_t maxMem, size_t nBest) {
  std::vector<const Transform *> *result = new std::vector<const Transform *>();
  size_t noOfTransforms = transforms->size();
  for (size_t i = 0; i < noOfTransforms; i++) {
    GeneralTransform gt = transforms->at(i);
    //		gt.print();
    if (Tristate::isNot(gt.isBatched)) {
      collapse(gt, false, gt.N, maxMem, result);
    }
    //		printf("\t---------------------------- end of noBatch \n");
    if (Tristate::is(gt.isBatched)) {
      collapseBatched(gt, maxMem, result);
    }
  }
  //	printf("\nabout to sort\n\n");
  std::sort(result->begin(), result->end(), perfSort);
  //	for (size_t i = 0; i < result->size(); i++) {
  //		result->at(i)->print();
  //	}
  while (result->size() > nBest) {
    delete result->back();
    result->pop_back();
  }
  return result;
}

void SizeOptimizer::collapseBatched(GeneralTransform &gt, size_t maxMem,
                                    std::vector<const Transform *> *result) {
  int lastN, currentN;
  lastN = currentN = 1;
  // double the amount of processed images, till you reach the limit
  bool tryNext = true;
  while (tryNext && (currentN <= gt.N)) {
    tryNext = collapse(gt, true, currentN, maxMem, result);
    if (tryNext) {
      lastN = currentN;
      currentN *= 2;
    }
  }
  // decrease by one till you find max
  currentN = std::min(gt.N, currentN - 1);
  tryNext = true;
  while (tryNext && (currentN > lastN)) {
    tryNext = !collapse(gt, true, currentN, maxMem, result);
    currentN--;
  }
}

bool SizeOptimizer::collapse(GeneralTransform &gt, bool isBatched, size_t N,
                             size_t maxMemMB,
                             std::vector<const Transform *> *result) {
  bool updated = false;

  std::vector<const Transform *> transforms;
  TransformGenerator::generate(gt.device, gt.X, gt.Y, gt.Z, N, isBatched,
                               gt.isFloat, gt.isForward, gt.isInPlace,
                               gt.isReal, transforms);

  size_t noOfTransforms = transforms.size();
  for (size_t i = 0; i < noOfTransforms; i++) {
    const Transform *t = transforms.at(i);
    BenchmarkResult r(t);
    cuFFTAdvisor::Benchmarker::estimatePlanSize(&r);
    size_t planSize = std::max(r.planSizeEstimateB, r.planSizeEstimate2B);
    size_t totalSizeBytes = r.transform->dataSizeB + planSize;
    size_t totalMB = std::ceil(toMB(totalSizeBytes));
    if (totalMB <= maxMemMB) {
      result->push_back(t);
      updated = true;
      r.transform = NULL;  // unbind, so that transform is not destroyed
    }                      // else 't' is deleted by destructor of 'r'
  }
  return updated;
}

size_t SizeOptimizer::getMaxSize(GeneralTransform &tr, int maxPercIncrease, bool squareOnly) {
  size_t maxXPow2 = std::ceil(log(tr.X) * log_2);
  size_t maxX = std::pow(2, maxXPow2);
  size_t maxYPow2 = squareOnly ? maxXPow2 : std::ceil(log(tr.Y) * log_2);
  size_t maxY = squareOnly ? maxX : std::pow(2, maxYPow2);
  size_t maxZPow2 = squareOnly ? maxXPow2 : std::ceil(log(tr.Z) * log_2);
  size_t maxZ = squareOnly? maxX : std::pow(2, maxZPow2);

  return std::min(maxX * maxY * maxZ,
                  (size_t)(tr.getDimSize() * ((maxPercIncrease / 100.f) + 1)));
}

std::vector<GeneralTransform> *SizeOptimizer::optimizeXYZ(GeneralTransform &tr,
                                                          size_t nBest,
                                                          int maxPercIncrease,
                                                          bool squareOnly) {
  std::vector<Polynom> *polysX = generatePolys(tr.X, tr.isFloat);
  std::vector<Polynom> *polysY;
  std::vector<Polynom> *polysZ;
  std::set<Polynom, valueComparator> *recPolysX = filterOptimal(polysX);
  std::set<Polynom, valueComparator> *recPolysY;
  std::set<Polynom, valueComparator> *recPolysZ;

  if ((tr.X == tr.Y)
      || (squareOnly && (tr.Y != 1))) {
    polysY = polysX;
    recPolysY = recPolysX;
  } else {
    polysY = generatePolys(tr.Y, tr.isFloat);
    recPolysY = filterOptimal(polysY);
  }

  if ((tr.X == tr.Z)
	  || (squareOnly && (tr.Z != 1))) {
    polysZ = polysX;
    recPolysZ = recPolysX;
  } else if (tr.Y == tr.Z) {
    polysZ = polysY;
    recPolysZ = recPolysY;
  } else {
    polysZ = generatePolys(tr.Z, tr.isFloat);
    recPolysZ = filterOptimal(polysZ);
  }

  size_t maxSize = getMaxSize(tr, maxPercIncrease, squareOnly);

  std::vector<GeneralTransform> *result = new std::vector<GeneralTransform>;
  std::set<Polynom>::iterator x;
  std::set<Polynom>::iterator y;
  std::set<Polynom>::iterator z;
  size_t found = 0;
  for (x = recPolysX->begin(); x != recPolysX->end(); ++x) {
    for (y = recPolysY->begin(); y != recPolysY->end(); ++y) {
      if (squareOnly && (x->value != y->value) && (y->value != 1)) continue;
      size_t xy = x->value * y->value;
      if (xy > maxSize)
        break;  // polynoms are sorted by size, we're already above the limit
      for (z = recPolysZ->begin(); z != recPolysZ->end(); ++z) {
        if (squareOnly && (x->value != z->value) && (z->value != 1)) continue;
        size_t xyz = xy * z->value;
        if ((found < nBest) && (xyz <= maxSize) && (xyz >= tr.getDimSize())) {
          // we can take nbest only, as others very probably won't be faster
          found++;
          GeneralTransform t((int)x->value, (int)y->value, (int)z->value, tr);
          result->push_back(t);
        }
      }
    }
  }

  if (polysZ != polysY) {
    delete polysZ;
    delete recPolysZ;
  }
  if (polysY != polysX) {
    delete polysY;
    delete recPolysY;
  }
  delete polysX;
  delete recPolysX;
  return result;
}

int SizeOptimizer::getNoOfPrimes(Polynom &poly) {
  int counter = 0;
  if (poly.exponent2 != 0) counter++;
  if (poly.exponent3 != 0) counter++;
  if (poly.exponent5 != 0) counter++;
  if (poly.exponent7 != 0) counter++;
  return counter;
}

int SizeOptimizer::getInvocations(int maxPower, size_t num) {
  int count = 0;
  while (0 != num) {
    for (size_t p = maxPower; p > 0; p--) {
      if (num >= p) {
        num -= p;
        count++;
        break;
      }
    }
  }
  return count;
}

int SizeOptimizer::getInvocationsV8(Polynom &poly, bool isFloat) {
  int result = 0;
  if (isFloat) {
    result += getInvocations(V8_RADIX_2_MAX_SP, poly.exponent2);
    result += getInvocations(V8_RADIX_3_MAX_SP, poly.exponent3);
    result += getInvocations(V8_RADIX_5_MAX_SP, poly.exponent5);
    result += getInvocations(V8_RADIX_7_MAX_SP, poly.exponent7);
  } else {
    result += getInvocations(V8_RADIX_2_MAX_DP, poly.exponent2);
    result += getInvocations(V8_RADIX_3_MAX_DP, poly.exponent3);
    result += getInvocations(V8_RADIX_5_MAX_DP, poly.exponent5);
    result += getInvocations(V8_RADIX_7_MAX_DP, poly.exponent7);
  }
  return result;
}

int SizeOptimizer::getInvocations(Polynom &poly, bool isFloat) {
  switch (version) {
    case (CudaVersion::V_8):
      return getInvocationsV8(poly, isFloat);
    //	case (CudaVersion::V_9):
    //		return getInvocationsV9(poly); // FIXME implement
    default:
      throw std::domain_error("Unsupported version of CUDA");
  }
}

std::vector<SizeOptimizer::Polynom> *SizeOptimizer::generatePolys(
    size_t num, bool isFloat) {
  std::vector<Polynom> *result = new std::vector<Polynom>();
  size_t maxPow2 = std::ceil(log(num) * log_2);
  size_t max = std::pow(2, maxPow2);
  size_t maxPow3 = std::ceil(std::log(max) * log_3);
  size_t maxPow5 = std::ceil(std::log(max) * log_5);
  size_t maxPow7 = std::ceil(std::log(max) * log_7);
  for (size_t a = 1; a <= maxPow2;
       a++) {  // we want at least one multiple of two
    for (size_t b = 0; b <= maxPow3; b++) {
      for (size_t c = 0; c <= maxPow5; c++) {
        for (size_t d = 0; d <= maxPow7; d++) {
          size_t value =
              std::pow(2, a) * std::pow(3, b) * std::pow(5, c) * std::pow(7, d);
          if ((value >= num) && (value <= max)) {
            Polynom p;
            p.value = value;
            p.exponent2 = a;
            p.exponent3 = b;
            p.exponent5 = c;
            p.exponent7 = d;
            p.invocations = getInvocations(p, isFloat);
            p.noOfPrimes = getNoOfPrimes(p);
            result->push_back(p);
          }
        }
      }
    }
  }
  return result;
}

std::set<SizeOptimizer::Polynom, SizeOptimizer::valueComparator>
    *SizeOptimizer::filterOptimal(std::vector<SizeOptimizer::Polynom> *input) {
  std::set<Polynom, valueComparator> *result =
      new std::set<Polynom, valueComparator>();
  // there are always polynoms with power of 2
  // we want polynoms where at least two other primes are zero
  size_t size = input->size();
  if (0 == size) {
    result->insert(UNIT);
    return result;
  }

  Polynom &minInv = input->at(0);
  Polynom &closest = minInv;
  for (size_t i = 1; i < size; i++) {
    Polynom &tmp = input->at(i);
    if (tmp.invocations < minInv.invocations) {
      // update min kernel invocations needed
      minInv = tmp;
    }
    if (closest.value > tmp.value) {
      closest = tmp;
    }
  }

  result->insert(closest);

  // filter
  for (size_t i = 0; i < size; i++) {
    Polynom &tmp = input->at(i);
    if ((tmp.invocations <= (minInv.invocations + 1)) &&
        (tmp.noOfPrimes <= 2)) {
      result->insert(tmp);
    }
  }
  return result;
}

}  // namespace cuFFTAdvisor
