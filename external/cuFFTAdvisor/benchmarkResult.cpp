#include "benchmarkResult.h"

namespace cuFFTAdvisor {

void BenchmarkResult::print(FILE* stream) const {
  if (NULL == stream) return;
  transform->print(stream);
  printf(
      "%s%.10f%s%.10f"  // plan estimation
      "%s%.10f%s%.10f"  // memory
      "%s%.10f%s%.10f%s%.10f"  // times
      "%s%lu%s%.10f%s%s",
      SEP, toMB(planSizeEstimateB), SEP, toMB(planSizeEstimate2B), SEP,
      toMB(planSizeRealB), SEP, toMB(transform->dataSizeB), SEP, planTimeMS,
      SEP, execTimeMS, SEP, totalTimeMS, SEP, transform->elems, SEP, getPerf(), SEP,
      errMsg.c_str());
}

void BenchmarkResult::printHeader(FILE* stream) {
  Transform::printHeader(stream);
  fprintf(
      stream,
      "%splan size estimate (MB)%splan size estimate 2 (MB)%splan size actual "
      "(MB)%sdata size (MB)"
      "%splan time (ms)%sexec time (ms/signal)%sexec time of batch (ms)"
      "%selements%sperformance (1k elem of signal/ms)%serror message",
      SEP, SEP, SEP, SEP, SEP, SEP, SEP, SEP, SEP, SEP);
}

bool BenchmarkResult::execSort(const BenchmarkResult* l,
                               const BenchmarkResult* r) {
  if (std::isnan(l->execTimeMS) && (!std::isnan(r->execTimeMS))) return false;
  if ((!std::isnan(l->execTimeMS)) && std::isnan(r->execTimeMS)) return true;
  return l->execTimeMS < r->execTimeMS;
}

inline float BenchmarkResult::getPerf() const {
  return std::isnan(this->planTimeMS)
             ? NAN
             : ((transform->elems / transform->N) / 1000.f) / this->execTimeMS;
}

}  // namespace cuFFTAdvisor
