#ifndef CUDA_FORWARD_ART_ZERNIKE3D_CU
#define CUDA_FORWARD_ART_ZERNIKE3D_CU

#include "cuda_forward_art_zernike3d.h"
#include "cuda_forward_art_zernike3d_defines.h"

namespace cuda_forward_art_zernike3D {

// Constants
static constexpr float CUDA_PI = 3.1415926535897f;
// Functions
#define SQRT sqrtf
#define ATAN2 atan2f
#define COS cosf
#define SIN sinf
#define POW powf
#define ABSC fabsf
#define CUDA_FLOOR floorf
#define CUDA_ROUND lroundf

#define IS_OUTSIDE2D(ImD, i, j) \
	((j) < STARTINGX((ImD)) || (j) > FINISHINGX((ImD)) || (i) < STARTINGY((ImD)) || (i) > FINISHINGY((ImD)))

// Smart casting to selected precision (at compile time)
// ...just shorter static_cast
#define CST(num) (static_cast<PrecisionType>((num)))

#define LIN_INTERP(a, l, h) ((l) + ((h) - (l)) * (a))

#define MODULO(a, b) ((a) - ((a) / (b) * (b)))

namespace device {

	template<typename PrecisionType>
	__forceinline__ __device__ PrecisionType ZernikeSphericalHarmonics(int l1,
																	   int n,
																	   int l2,
																	   int m,
																	   PrecisionType xr,
																	   PrecisionType yr,
																	   PrecisionType zr,
																	   PrecisionType rr)
	{
		// General variables
		PrecisionType r2 = rr * rr, xr2 = xr * xr, yr2 = yr * yr, zr2 = zr * zr;

#if L2 >= 5
		// Variables needed for l2 >= 5
		PrecisionType tht = CST(0.0), phi = CST(0.0), cost = CST(0.0), sinp = CST(0.0), cost2 = CST(0.0),
					  sinp2 = CST(0.0), cosp = CST(0.0);
		if (l2 >= 5) {
			PrecisionType mf = CST(m);
			tht = ATAN2(yr, xr);
			phi = ATAN2(zr, SQRT(xr2 + yr2));
			sinp = SIN(ABSC(mf) * phi);
			cost = COS(tht);
			cosp = COS(ABSC(mf) * phi);
			sinp2 = sinp * sinp;
			cost2 = cost * cost;
		}
#endif	// L2 >= 5

		// Zernike polynomial
		PrecisionType R = CST(0.0);

		switch (l1) {
			case 0:
				R = SQRT(CST(3.0));
				break;
			case 1:
				R = SQRT(CST(5.0)) * rr;
				break;
			case 2:
				switch (n) {
					case 0:
						R = CST(-0.5) * SQRT(CST(7.0)) * (CST(2.5) * (CST(1.0) - CST(2.0) * r2) + CST(0.5));
						break;
					case 2:
						R = SQRT(CST(7.0)) * r2;
						break;
				}
				break;
#if L1 >= 3
			case 3:
				switch (n) {
					case 1:
						R = CST(-1.5) * rr * (CST(3.5) * (CST(1.0) - CST(2.0) * r2) + CST(1.5));
						break;
					case 3:
						R = CST(3.0) * r2 * rr;
				}
				break;
#endif	// L1 >= 3
#if L1 >= 4
			case 4:
				switch (n) {
					case 0:
						R = SQRT(CST(11.0))
							* ((CST(63.0) * r2 * r2 / CST(8.0)) - (CST(35.0) * r2 / CST(4.0)) + (CST(15.0) / CST(8.0)));
						break;
					case 2:
						R = CST(-0.5) * SQRT(CST(11.0)) * r2 * (CST(4.5) * (CST(1.0) - CST(2.0) * r2) + CST(2.5));
						break;
					case 4:
						R = SQRT(CST(11.0)) * r2 * r2;
						break;
				}
				break;
#endif	// L1 >= 4
#if L1 >= 5
			case 5:
				switch (n) {
					case 1:
						R = SQRT(CST(13.0)) * rr
							* ((CST(99.0) * r2 * r2 / CST(8.0)) - (CST(63.0) * r2 / CST(4.0)) + (CST(35.0) / CST(8.0)));
						break;
					case 3:
						R = CST(-0.5) * SQRT(CST(13.0)) * r2 * rr * (CST(5.5) * (CST(1.0) - CST(2.0) * r2) + CST(3.5));
						break;
					case 5:
						R = SQRT(CST(13.0)) * r2 * r2 * rr;
						break;
				}
				break;
#endif	// L1 >= 5
#if L1 >= 6
			case 6:
				switch (n) {
					case 0:
						R = CST(103.8) * r2 * r2 * r2 - CST(167.7) * r2 * r2 + CST(76.25) * r2 - CST(8.472);
						break;
					case 2:
						R = CST(69.23) * r2 * r2 * r2 - CST(95.86) * r2 * r2 + CST(30.5) * r2;
						break;
					case 4:
						R = CST(25.17) * r2 * r2 * r2 - CST(21.3) * r2 * r2;
						break;
					case 6:
						R = CST(3.873) * r2 * r2 * r2;
						break;
				}
				break;
#endif	// L1 >= 6
#if L1 >= 7
			case 7:
				switch (n) {
					case 1:
						R = CST(184.3) * r2 * r2 * r2 * rr - CST(331.7) * r2 * r2 * rr + CST(178.6) * r2 * rr
							- CST(27.06) * rr;
						break;
					case 3:
						R = CST(100.5) * r2 * r2 * r2 * rr - CST(147.4) * r2 * r2 * rr + CST(51.02) * r2 * rr;
						break;
					case 5:
						R = CST(30.92) * r2 * r2 * r2 * rr - CST(26.8) * r2 * r2 * rr;
						break;
					case 7:
						R = CST(4.123) * r2 * r2 * r2 * rr;
						break;
				}
				break;
#endif	// L1 >= 7
		}

		// Spherical harmonic
		PrecisionType Y = CST(0.0);

		switch (l2) {
			case 0:
				Y = (CST(1.0) / CST(2.0)) * SQRT((PrecisionType)CST(1.0) / CUDA_PI);
				break;
			case 1:
				switch (m) {
					case -1:
						Y = SQRT(CST(3.0) / (CST(4.0) * CUDA_PI)) * yr;
						break;
					case 0:
						Y = SQRT(CST(3.0) / (CST(4.0) * CUDA_PI)) * zr;
						break;
					case 1:
						Y = SQRT(CST(3.0) / (CST(4.0) * CUDA_PI)) * xr;
						break;
				}
				break;
			case 2:
				switch (m) {
					case -2:
						Y = SQRT(CST(15.0) / (CST(4.0) * CUDA_PI)) * xr * yr;
						break;
					case -1:
						Y = SQRT(CST(15.0) / (CST(4.0) * CUDA_PI)) * zr * yr;
						break;
					case 0:
						Y = SQRT(CST(5.0) / (CST(16.0) * CUDA_PI)) * (-xr2 - yr2 + CST(2.0) * zr2);
						break;
					case 1:
						Y = SQRT(CST(15.0) / (CST(4.0) * CUDA_PI)) * xr * zr;
						break;
					case 2:
						Y = SQRT(CST(15.0) / (CST(16.0) * CUDA_PI)) * (xr2 - yr2);
						break;
				}
				break;
#if L2 >= 3
			case 3:
				switch (m) {
					case -3:
						Y = SQRT(CST(35.0) / (CST(16.0) * CST(2.0) * CUDA_PI)) * yr * (CST(3.0) * xr2 - yr2);
						break;
					case -2:
						Y = SQRT(CST(105.0) / (CST(4.0) * CUDA_PI)) * zr * yr * xr;
						break;
					case -1:
						Y = SQRT(CST(21.0) / (CST(16.0) * CST(2.0) * CUDA_PI)) * yr * (CST(4.0) * zr2 - xr2 - yr2);
						break;
					case 0:
						Y = SQRT(CST(7.0) / (CST(16.0) * CUDA_PI)) * zr
							* (CST(2.0) * zr2 - CST(3.0) * xr2 - CST(3.0) * yr2);
						break;
					case 1:
						Y = SQRT(CST(21.0) / (CST(16.0) * CST(2.0) * CUDA_PI)) * xr * (CST(4.0) * zr2 - xr2 - yr2);
						break;
					case 2:
						Y = SQRT(CST(105.0) / (CST(16.0) * CUDA_PI)) * zr * (xr2 - yr2);
						break;
					case 3:
						Y = SQRT(CST(35.0) / (CST(16.0) * CST(2.0) * CUDA_PI)) * xr * (xr2 - CST(3.0) * yr2);
						break;
				}
				break;
#endif	// L2 >= 3
#if L2 >= 4
			case 4:
				switch (m) {
					case -4:
						Y = SQRT((CST(35.0) * CST(9.0)) / (CST(16.0) * CUDA_PI)) * yr * xr * (xr2 - yr2);
						break;
					case -3:
						Y = SQRT((CST(9.0) * CST(35.0)) / (CST(16.0) * CST(2.0) * CUDA_PI)) * yr * zr
							* (CST(3.0) * xr2 - yr2);
						break;
					case -2:
						Y = SQRT((CST(9.0) * CST(5.0)) / (CST(16.0) * CUDA_PI)) * yr * xr
							* (CST(7.0) * zr2 - (xr2 + yr2 + zr2));
						break;
					case -1:
						Y = SQRT((CST(9.0) * CST(5.0)) / (CST(16.0) * CST(2.0) * CUDA_PI)) * yr * zr
							* (CST(7.0) * zr2 - CST(3.0) * (xr2 + yr2 + zr2));
						break;
					case 0:
						Y = SQRT(CST(9.0) / (CST(16.0) * CST(16.0) * CUDA_PI))
							* (CST(35.0) * zr2 * zr2 - CST(30.0) * zr2 + CST(3.0));
						break;
					case 1:
						Y = SQRT((CST(9.0) * CST(5.0)) / (CST(16.0) * CST(2.0) * CUDA_PI)) * xr * zr
							* (CST(7.0) * zr2 - CST(3.0) * (xr2 + yr2 + zr2));
						break;
					case 2:
						Y = SQRT((CST(9.0) * CST(5.0)) / (CST(8.0) * CST(8.0) * CUDA_PI)) * (xr2 - yr2)
							* (CST(7.0) * zr2 - (xr2 + yr2 + zr2));
						break;
					case 3:
						Y = SQRT((CST(9.0) * CST(35.0)) / (CST(16.0) * CST(2.0) * CUDA_PI)) * xr * zr
							* (xr2 - CST(3.0) * yr2);
						break;
					case 4:
						Y = SQRT((CST(9.0) * CST(35.0)) / (CST(16.0) * CST(16.0) * CUDA_PI))
							* (xr2 * (xr2 - CST(3.0) * yr2) - yr2 * (CST(3.0) * xr2 - yr2));
						break;
				}
				break;
#endif	// L2 >= 4
#if L2 >= 5
			case 5:
				switch (m) {
					case -5:
						Y = (CST(3.0) / CST(16.0)) * SQRT(CST(77.0) / (CST(2.0) * CUDA_PI)) * sinp2 * sinp2 * sinp
							* SIN(CST(5.0) * phi);
						break;
					case -4:
						Y = (CST(3.0) / CST(8.0)) * SQRT(CST(385.0) / (CST(2.0) * CUDA_PI)) * sinp2 * sinp2
							* SIN(CST(4.0) * phi);
						break;
					case -3:
						Y = (CST(1.0) / CST(16.0)) * SQRT(CST(385.0) / (CST(2.0) * CUDA_PI)) * sinp2 * sinp
							* (CST(9.0) * cost2 - CST(1.0)) * SIN(CST(3.0) * phi);
						break;
					case -2:
						Y = (CST(1.0) / CST(4.0)) * SQRT(CST(1155.0) / (CST(4.0) * CUDA_PI)) * sinp2
							* (CST(3.0) * cost2 * cost - cost) * SIN(CST(2.0) * phi);
						break;
					case -1:
						Y = (CST(1.0) / CST(8.0)) * SQRT(CST(165.0) / (CST(4.0) * CUDA_PI)) * sinp
							* (CST(21.0) * cost2 * cost2 - CST(14.0) * cost2 + CST(1.0)) * SIN(phi);
						break;
					case 0:
						Y = (CST(1.0) / CST(16.0)) * SQRT(CST(11.0) / CUDA_PI)
							* (CST(63.0) * cost2 * cost2 * cost - CST(70.0) * cost2 * cost + CST(15.0) * cost);
						break;
					case 1:
						Y = (CST(1.0) / CST(8.0)) * SQRT(CST(165.0) / (CST(4.0) * CUDA_PI)) * sinp
							* (CST(21.0) * cost2 * cost2 - CST(14.0) * cost2 + CST(1.0)) * COS(phi);
						break;
					case 2:
						Y = (CST(1.0) / CST(4.0)) * SQRT(CST(1155.0) / (CST(4.0) * CUDA_PI)) * sinp2
							* (CST(3.0) * cost2 * cost - cost) * COS(CST(2.0) * phi);
						break;
					case 3:
						Y = (CST(1.0) / CST(16.0)) * SQRT(CST(385.0) / (CST(2.0) * CUDA_PI)) * sinp2 * sinp
							* (CST(9.0) * cost2 - CST(1.0)) * COS(CST(3.0) * phi);
						break;
					case 4:
						Y = (CST(3.0) / CST(8.0)) * SQRT(CST(385.0) / (CST(2.0) * CUDA_PI)) * sinp2 * sinp2
							* COS(CST(4.0) * phi);
						break;
					case 5:
						Y = (CST(3.0) / CST(16.0)) * SQRT(CST(77.0) / (CST(2.0) * CUDA_PI)) * sinp2 * sinp2 * sinp
							* COS(CST(5.0) * phi);
						break;
				}
				break;
#endif	// L2 >= 5
#if L2 >= 6
			case 6:
				switch (m) {
					case -6:
						Y = -CST(0.6832) * sinp * POW(cost2 - CST(1.0), CST(3.0));
						break;
					case -5:
						Y = CST(2.367) * cost * sinp * POW(CST(1.0) - CST(1.0) * cost2, CST(2.5));
						break;
					case -4:
						Y = CST(0.001068) * sinp * (CST(5198.0) * cost2 - CST(472.5)) * POW(cost2 - CST(1.0), CST(2.0));
						break;
					case -3:
						Y = -CST(0.005849) * sinp * POW(CST(1.0) - CST(1.0) * cost2, CST(1.5))
							* (-CST(1732.0) * cost2 * cost + CST(472.5) * cost);
						break;
					case -2:
						Y = -CST(0.03509) * sinp * (cost2 - CST(1.0))
							* (CST(433.1) * cost2 * cost2 - CST(236.2) * cost2 + CST(13.12));
						break;
					case -1:
						Y = CST(0.222) * sinp * POW(CST(1.0) - CST(1.0) * cost2, CST(0.5))
							* (CST(86.62) * cost2 * cost2 * cost - CST(78.75) * cost2 * cost + CST(13.12) * cost);
						break;
					case 0:
						Y = CST(14.68) * cost2 * cost2 * cost2 - CST(20.02) * cost2 * cost2 + CST(6.675) * cost2
							- CST(0.3178);
						break;
					case 1:
						Y = CST(0.222) * cosp * POW(CST(1.0) - CST(1.0) * cost2, CST(0.5))
							* (CST(86.62) * cost2 * cost2 * cost - CST(78.75) * cost2 * cost
							   + CST(13.12) * cost);  //FIXME: Here we need cost instead of cost2
						break;
					case 2:
						Y = -CST(0.03509) * cosp * (cost2 - CST(1.0))
							* (CST(433.1) * cost2 * cost2 - CST(236.2) * cost2 + CST(13.12));
						break;
					case 3:
						Y = -CST(0.005849) * cosp * POW(CST(1.0) - CST(1.0) * cost2, CST(1.5))
							* (-CST(1732.0) * cost2 * cost + CST(472.5) * cost);
						break;
					case 4:
						Y = CST(0.001068) * cosp * (CST(5198.0) * cost2 - CST(472.5)) * POW(cost2 - CST(1.0), CST(2.0));
						break;
					case 5:
						Y = CST(2.367) * cost * cosp * POW(CST(1.0) - CST(1.0) * cost2, CST(2.5));
						break;
					case 6:
						Y = -CST(0.6832) * cosp * POW(cost2 - CST(1.0), CST(3.0));
						break;
				}
				break;
#endif	// L2 >= 6
#if L2 >= 7
			case 7:
				switch (m) {
					case -7:
						Y = CST(0.7072) * sinp * POW(CST(1.0) - CST(1.0) * cost2, CST(3.5));
						break;
					case -6:
						Y = -CST(2.646) * cost * sinp * POW(cost2 - CST(1.0), CST(3.0));
						break;
					case -5:
						Y = CST(9.984e-5) * sinp * POW(CST(1.0) - CST(1.0) * cost2, CST(2.5))
							* (CST(67570.0) * cost2 - CST(5198.0));
						break;
					case -4:
						Y = -CST(0.000599) * sinp * POW(cost2 - CST(1.0), CST(2.0))
							* (-CST(22520.0) * cost2 * cost + CST(5198.0) * cost);
						break;
					case -3:
						Y = CST(0.003974) * sinp * POW(CST(1.0) - CST(1.0) * cost2, CST(1.5))
							* (CST(5631.0) * cost2 * cost2 - CST(2599.0) * cost2 + CST(118.1));
						break;
					case -2:
						Y = -CST(0.0281) * sinp * (cost2 - CST(1.0))
							* (CST(1126.0) * cost2 * cost2 * cost - CST(866.2) * cost2 * cost + CST(118.1) * cost);
						break;
					case -1:
						Y = CST(0.2065) * sinp * POW(CST(1.0) - CST(1.0) * cost2, CST(0.5))
							* (CST(187.7) * cost2 * cost2 * cost2 - CST(216.6) * cost2 * cost2 + CST(59.06) * cost2
							   - CST(2.188));
						break;
					case 0:
						Y = CST(29.29) * cost2 * cost2 * cost2 * cost - CST(47.32) * cost2 * cost2 * cost
							+ CST(21.51) * cost2 * cost - CST(2.39) * cost;
						break;
					case 1:
						Y = CST(0.2065) * cosp * POW(CST(1.0) - CST(1.0) * cost2, CST(0.5))
							* (CST(187.7) * cost2 * cost2 * cost2 - CST(216.6) * cost2 * cost2 + CST(59.06) * cost2
							   - CST(2.188));
						break;
					case 2:
						Y = -CST(0.0281) * cosp * (cost2 - CST(1.0))
							* (CST(1126.0) * cost2 * cost2 * cost - CST(866.2) * cost2 * cost + CST(118.1) * cost);
						break;
					case 3:
						Y = CST(0.003974) * cosp * POW(CST(1.0) - CST(1.0) * cost2, CST(1.5))
							* (CST(5631.0) * cost2 * cost2 - CST(2599.0) * cost2 + CST(118.1));
						break;
					case 4:
						Y = -CST(0.000599) * cosp * POW(cost2 - CST(1.0), CST(2.0))
							* (-CST(22520.0) * cost2 * cost + CST(5198.0) * cost);
						break;
					case 5:
						Y = CST(9.984e-5) * cosp * POW(CST(1.0) - CST(1.0) * cost2, CST(2.5))
							* (CST(67570.0) * cost2 - CST(5198.0));
						break;
					case 6:
						Y = -CST(2.646) * cost * cosp * POW(cost2 - CST(1.0), CST(3.0));
						break;
					case 7:
						Y = CST(0.7072) * cosp * POW(CST(1.0) - CST(1.0) * cost2, CST(3.5));
						break;
				}
				break;
#endif	// L2 >= 7
		}

		return R * Y;
	}

	template<typename PrecisionType>
	__device__ PrecisionType atomicAddPrecision(PrecisionType *addr, PrecisionType val)
	{
		return atomicAdd(addr, val);
	}

	template<>
	__device__ double atomicAddPrecision(double *address, double val)
	{
		unsigned long long int *address_as_ull = (unsigned long long int *)address;
		unsigned long long int old = *address_as_ull, assumed;

		do {
			assumed = old;
			old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

			// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
		} while (assumed != old);

		return __longlong_as_double(old);
	}

	template<typename PrecisionType>
	__device__ void splattingAtPos(const PrecisionType pos_x,
								   const PrecisionType pos_y,
								   MultidimArrayCuda<PrecisionType> &mP,
								   MultidimArrayCuda<PrecisionType> &mW,
								   const PrecisionType weight)
	{
		int i = static_cast<int>(CUDA_ROUND(pos_y));
		int j = static_cast<int>(CUDA_ROUND(pos_x));
		if (!IS_OUTSIDE2D(mP, i, j)) {
			atomicAddPrecision(&A2D_ELEM(mP, i, j), weight);
			atomicAddPrecision(&A2D_ELEM(mW, i, j), CST(1.0));
		}
	}

	template<typename PrecisionType>
	__device__ size_t findCuda(const PrecisionType *begin, const size_t size, PrecisionType value)
	{
		if (size <= 0) {
			return 0;
		}
		for (size_t i = 0; i < size; i++) {
			if (begin[i] == value) {
				return i;
			}
		}
		return size;
	}

	template<typename PrecisionType>
	__device__ PrecisionType interpolatedElement2DCuda(PrecisionType x,
													   PrecisionType y,
													   const MultidimArrayCuda<PrecisionType> &diffImage)
	{
		int x0 = CUDA_FLOOR(x);
		PrecisionType fx = x - x0;
		int x1 = x0 + 1;
		int y0 = CUDA_FLOOR(y);
		PrecisionType fy = y - y0;
		int y1 = y0 + 1;

		int i0 = STARTINGY(diffImage);
		int j0 = STARTINGX(diffImage);
		int iF = FINISHINGY(diffImage);
		int jF = FINISHINGX(diffImage);

#define ASSIGNVAL2DCUDA(d, i, j)                      \
	if ((j) < j0 || (j) > jF || (i) < i0 || (i) > iF) \
		d = (PrecisionType)0;                         \
	else                                              \
		d = A2D_ELEM(diffImage, i, j);

		PrecisionType d00, d10, d11, d01;
		ASSIGNVAL2DCUDA(d00, y0, x0);
		ASSIGNVAL2DCUDA(d01, y0, x1);
		ASSIGNVAL2DCUDA(d10, y1, x0);
		ASSIGNVAL2DCUDA(d11, y1, x1);

		PrecisionType d0 = LIN_INTERP(fx, d00, d01);
		PrecisionType d1 = LIN_INTERP(fx, d10, d11);
		return LIN_INTERP(fy, d0, d1);
	}

}  // namespace device

/*
 * The first beast
 */
template<typename PrecisionType, bool usesZernike>
__global__ void forwardKernel(const MultidimArrayCuda<PrecisionType> cudaMV,
							  const int *cudaVRecMaskF,
							  const unsigned *cudaCoordinatesF,
							  const int xdim,
							  const int ydim,
							  const unsigned sizeF,
							  MultidimArrayCuda<PrecisionType> *cudaP,
							  MultidimArrayCuda<PrecisionType> *cudaW,
							  const unsigned sigma_size,
							  const PrecisionType *cudaSigma,
							  const PrecisionType iRmaxF,
							  const unsigned idxY0,
							  const unsigned idxZ0,
							  const int *cudaVL1,
							  const int *cudaVN,
							  const int *cudaVL2,
							  const int *cudaVM,
							  const PrecisionType *cudaClnm,
							  const PrecisionType r0,
							  const PrecisionType r1,
							  const PrecisionType r2,
							  const PrecisionType r3,
							  const PrecisionType r4,
							  const PrecisionType r5)
{
	int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (sizeF <= threadIndex) {
		return;
	}
	unsigned threadPosition = cudaCoordinatesF[threadIndex];
	int img_idx = 0;
	if (sigma_size > 1) {
		PrecisionType sigma_mask = cudaVRecMaskF[threadIndex];
		img_idx = device::findCuda(cudaSigma, sigma_size, sigma_mask);
	}
	auto &mP = cudaP[img_idx];
	auto &mW = cudaW[img_idx];
	__builtin_assume(xdim > 0);
	__builtin_assume(ydim > 0);
	unsigned cubeX = threadPosition % xdim;
	unsigned cubeY = threadPosition / xdim % ydim;
	unsigned cubeZ = threadPosition / (xdim * ydim);
	int k = STARTINGZ(cudaMV) + cubeZ;
	int i = STARTINGY(cudaMV) + cubeY;
	int j = STARTINGX(cudaMV) + cubeX;
	PrecisionType weight = A3D_ELEM(cudaMV, k, i, j);
	PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
	if (usesZernike) {
		auto k2 = k * k;
		auto kr = k * iRmaxF;
		auto k2i2 = k2 + i * i;
		auto ir = i * iRmaxF;
		auto r2 = k2i2 + j * j;
		auto jr = j * iRmaxF;
		auto rr = SQRT(r2) * iRmaxF;
		for (size_t idx = 0; idx < idxY0; idx++) {
			auto l1 = cudaVL1[idx];
			auto n = cudaVN[idx];
			auto l2 = cudaVL2[idx];
			auto m = cudaVM[idx];
			if (rr > 0 || l2 == 0) {
				PrecisionType zsph = device::ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
				gx += cudaClnm[idx] * (zsph);
				gy += cudaClnm[idx + idxY0] * (zsph);
				gz += cudaClnm[idx + idxZ0] * (zsph);
			}
		}
	}

	auto r_x = j + gx;
	auto r_y = i + gy;
	auto r_z = k + gz;

	auto pos_x = r0 * r_x + r1 * r_y + r2 * r_z;
	auto pos_y = r3 * r_x + r4 * r_y + r5 * r_z;
	device::splattingAtPos(pos_x, pos_y, mP, mW, weight);
}

/*
 * The second beast
 */
template<typename PrecisionType, bool usesZernike>
__global__ void backwardKernel(MultidimArrayCuda<PrecisionType> cudaMV,
							   const MultidimArrayCuda<PrecisionType> cudaMId,
							   const MultidimArrayCuda<int> VRecMaskB,
							   const int lastZ,
							   const int lastY,
							   const int lastX,
							   const int step,
							   const PrecisionType iRmaxF,
							   const size_t idxY0,
							   const size_t idxZ0,
							   const int *cudaVL1,
							   const int *cudaVN,
							   const int *cudaVL2,
							   const int *cudaVM,
							   const PrecisionType *cudaClnm,
							   const PrecisionType *cudaR)
{
	int cubeX = threadIdx.x + blockIdx.x * blockDim.x;
	int cubeY = threadIdx.y + blockIdx.y * blockDim.y;
	int cubeZ = threadIdx.z + blockIdx.z * blockDim.z;
	int k = STARTINGZ(cudaMV) + cubeZ;
	int i = STARTINGY(cudaMV) + cubeY;
	int j = STARTINGX(cudaMV) + cubeX;
	PrecisionType gx = 0.0, gy = 0.0, gz = 0.0;
	if (A3D_ELEM(VRecMaskB, k, i, j) != 0) {
		if (usesZernike) {
			auto k2 = k * k;
			auto kr = k * iRmaxF;
			auto k2i2 = k2 + i * i;
			auto ir = i * iRmaxF;
			auto r2 = k2i2 + j * j;
			auto jr = j * iRmaxF;
			auto rr = SQRT(r2) * iRmaxF;
			for (size_t idx = 0; idx < idxY0; idx++) {
				auto l1 = cudaVL1[idx];
				auto n = cudaVN[idx];
				auto l2 = cudaVL2[idx];
				auto m = cudaVM[idx];
				if (rr > 0 || l2 == 0) {
					PrecisionType zsph = device::ZernikeSphericalHarmonics(l1, n, l2, m, jr, ir, kr, rr);
					gx += cudaClnm[idx] * (zsph);
					gy += cudaClnm[idx + idxY0] * (zsph);
					gz += cudaClnm[idx + idxZ0] * (zsph);
				}
			}
		}

		auto r_x = j + gx;
		auto r_y = i + gy;
		auto r_z = k + gz;

		auto pos_x = cudaR[0] * r_x + cudaR[1] * r_y + cudaR[2] * r_z;
		auto pos_y = cudaR[3] * r_x + cudaR[4] * r_y + cudaR[5] * r_z;
		PrecisionType voxel = device::interpolatedElement2DCuda(pos_x, pos_y, cudaMId);
		A3D_ELEM(cudaMV, k, i, j) += voxel;
	}
}
}  // namespace cuda_forward_art_zernike3D
#endif	//CUDA_FORWARD_ART_ZERNIKE3D_CU
