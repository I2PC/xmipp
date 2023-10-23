#include <ostream>
#include "data/fft_settings.h"

template<typename T>
std::ostream &operator<<(std::ostream &os,
                         const FFTSettings<T> &s)
{
    os << s.m_spatial.x() << "(" << s.m_freq.x() << ")"
       << " * " << s.m_spatial.y() << " * "
       << s.m_spatial.z() << " * " << s.m_spatial.n() << ", batch: " << s.m_batch
       << ", inPlace: " << (s.m_isInPlace ? "yes" : "no") 
       << ", isForward: " << (s.m_isForward ? "yes" : "no");
    return os;
}

// explicit instantiation
template std::ostream& operator<< <float>(std::ostream&, FFTSettings<float> const&);
template std::ostream& operator<< <double>(std::ostream&, FFTSettings<double> const&);