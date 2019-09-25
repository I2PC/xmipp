namespace Gpu {

template< typename T >
__device__
T interp(T x, T x0, T y0, T xF, T yF) {
	return y0 + ((x - x0) * (yF - y0)) / (xF - x0);
}

template< typename T >
__device__
T getCDFProbability(T xi, const T* x, const T* probXLessThanx, size_t Nsteps, T minVal, T maxVal) {
	if (xi > maxVal)
		return 1;
	else if (xi < minVal)
		return 0;
	else {
		size_t N = Nsteps;
		if (xi < x[0])
			return interp(xi, minVal, static_cast<T>(0.0), x[0], probXLessThanx[0]);
		else if (xi > x[N - 1])
			return interp(xi, x[N - 1], probXLessThanx[N - 1], maxVal, static_cast<T>(1.0));
		else
		{
			int iLeft = 0;
			int iRight = N - 1;
			while (iLeft <= iRight)
			{
				int iMiddle = iLeft + (iRight - iLeft)/2;
				if (xi >= x[iMiddle] && xi <= x[iMiddle + 1])
				{
					if (x[iMiddle] == x[iMiddle + 1])
						return 0.5 * (probXLessThanx[iMiddle] + probXLessThanx[iMiddle + 1]);
					else
						return interp(xi, x[iMiddle], probXLessThanx[iMiddle],
										 x[iMiddle + 1], probXLessThanx[iMiddle + 1]);
				}
				else if (xi < x[iMiddle])
					iRight = iMiddle;
				else
					iLeft = iMiddle;
			}
		}
	}
    return 0;
}

} // namespace Gpu
