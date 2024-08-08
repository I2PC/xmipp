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
#if L1 >= 8
			case 8:
				switch (n)
				{
				case 0:
					R = CST(413.9)*r2*r2*r2*r2 - CST(876.5)*r2*r2*r2 + CST(613.6)*r2*r2 - CST(157.3)*r2 + CST(10.73);
					break;
				case 2:
					R = CST(301.0)*r2*r2*r2*r2 - CST(584.4)*r2*r2*r2 + CST(350.6)*r2*r2 - CST(62.93)*r2;
					break;
				case 4:
					R = CST(138.9)*r2*r2*r2*r2 - CST(212.5)*r2*r2*r2 + CST(77.92)*r2*r2;
					break;
				case 6:
					R = CST(37.05)*r2*r2*r2*r2 - CST(32.69)*r2*r2*r2;
					break;
				case 8:
					R = CST(4.359)*r2*r2*r2*r2;
					break;
				} 
				break;
#endif // L1 >= 8
#if L1 >= 9
			case 9:
				switch (n)
				{
				case 1:
					R = CST(751.6)*r2*r2*r2*r2*rr - CST(1741.0)*r2*r2*r2*rr + CST(1382.0)*r2*r2*rr - CST(430.0)*r2*rr + CST(41.35)*rr;
					break;
				case 3:
					R = CST(462.6)*r2*r2*r2*r2*rr - CST(949.5)*r2*r2*r2*rr + CST(614.4)*r2*r2*rr - CST(122.9)*r2*rr;
					break;
				case 5:
					R = CST(185.0)*r2*r2*r2*r2*rr - CST(292.1)*r2*r2*r2*rr + CST(111.7)*r2*r2*rr;
					break;
				case 7:
					R = CST(43.53)*r2*r2*r2*r2*rr - CST(38.95)*r2*r2*r2*rr;
					break;
				case 9:
					R = CST(4.583)*r2*r2*r2*r2*rr;
					break;
				} 
				break;
#endif // L1 >= 9
#if L1 >= 10
			case 10:
				switch (n)
				{
				case 0:
					R = CST(1652.0)*r2*r2*r2*r2*r2 - CST(4326.0)*r2*r2*r2*r2 + CST(4099.0)*r2*r2*r2 - CST(1688.0)*r2*r2 + CST(281.3)*r2 - CST(12.98);
					break;
				case 2:
					R = CST(1271.0)*r2*r2*r2*r2*r2 - CST(3147.0)*r2*r2*r2*r2 + CST(2732.0)*r2*r2*r2 - CST(964.4)*r2*r2 + CST(112.5)*r2;
					break;
				case 4:
					R = CST(677.7)*r2*r2*r2*r2*r2 - CST(1452.0)*r2*r2*r2*r2 + CST(993.6)*r2*r2*r2 - CST(214.3)*r2*r2;
					break;
				case 6:
					R = CST(239.2)*r2*r2*r2*r2*r2 - CST(387.3)*r2*r2*r2*r2 + CST(152.9)*r2*r2*r2;
					break;
				case 8:
					R = CST(50.36)*r2*r2*r2*r2*r2 - CST(45.56)*r2*r2*r2*r2;
					break;
				case 10:
					R = CST(4.796)*r2*r2*r2*r2*r2;
					break;
				} 
				break;
#endif // L1 >= 10
# if L1 >= 11
			case 11:
				switch (n)
				{
				case 1:
					R = rr*-CST(5.865234375E+1)+(r2*rr)*CST(8.7978515625E+2)-(r2*r2*rr)*CST(4.2732421875E+3)+(r2*r2*r2*rr)*CST(9.0212890625E+3)-(r2*r2*r2*r2*rr)*CST(8.61123046875E+3)+POW(rr,CST(1.1E+1))*CST(3.04705078125E+3);
					break;
				case 3:
					R = (r2*rr)*CST(2.513671875E+2)-(r2*r2*rr)*CST(1.89921875E+3)+(r2*r2*r2*rr)*CST(4.920703125E+3)-(r2*r2*r2*r2*rr)*CST(5.29921875E+3)+POW(rr,CST(1.1E+1))*CST(2.0313671875E+3);
					break;
				case 5:
					R = (r2*r2*rr)*-CST(3.453125E+2)+(r2*r2*r2*rr)*CST(1.5140625E+3)-(r2*r2*r2*r2*rr)*CST(2.1196875E+3)+POW(rr,CST(1.1E+1))*CST(9.559375E+2);
					break;
				case 7:
					R = (r2*r2*r2*rr)*CST(2.01875E+2)-(r2*r2*r2*r2*rr)*CST(4.9875E+2)+POW(rr,CST(1.1E+1))*CST(3.01875E+2);
					break;
				case 9:
					R = (r2*r2*r2*r2*rr)*-CST(5.25E+1)+POW(rr,CST(1.1E+1))*CST(5.75E+1);
					break;
				case 11:
					R = POW(rr,CST(1.1E+1))*CST(5.0);
					break;
				} 
				break;
#endif // L1 >= 11
#if L1 >= 12
		case 12:
			switch (n)
			{
			case 0:
				R = (r2)*-CST(4.57149777110666E+2)+(r2*r2)*CST(3.885773105442524E+3)-(r2*r2*r2)*CST(1.40627979054153E+4)+(r2*r2*r2*r2)*CST(2.460989633446932E+4)-POW(rr,CST(1.0E+1))*CST(2.05828223888278E+4)+POW(rr,CST(1.2E+1))*CST(6.597058457955718E+3)+CST(1.523832590368693E+1);
				break;
			case 2:
				R = (r2)*-CST(1.828599108443595E+2)+(r2*r2)*CST(2.220441774539649E+3)-(r2*r2*r2)*CST(9.375198603600264E+3)+(r2*r2*r2*r2)*CST(1.789810642504692E+4)-POW(rr,CST(1.0E+1))*CST(1.583294029909372E+4)+POW(rr,CST(1.2E+1))*CST(5.277646766364574E+3);
				break;
			case 4:
				R = (r2*r2)*CST(4.934315054528415E+2)-(r2*r2*r2)*CST(3.409163128584623E+3)+(r2*r2*r2*r2)*CST(8.260664503872395E+3)-POW(rr,CST(1.0E+1))*CST(8.444234826177359E+3)+POW(rr,CST(1.2E+1))*CST(3.104498097866774E+3);
				break;
			case 6:
				R = (r2*r2*r2)*-CST(5.244866351671517E+2)+(r2*r2*r2*r2)*CST(2.202843867704272E+3)-POW(rr,CST(1.0E+1))*CST(2.98031817394495E+3)+POW(rr,CST(1.2E+1))*CST(1.307157093837857E+3);
				break;
			case 8:
				R = (r2*r2*r2*r2)*CST(2.591581020820886E+2)-POW(rr,CST(1.0E+1))*CST(6.274354050420225E+2)+POW(rr,CST(1.2E+1))*CST(3.734734553815797E+2);
				break;
			case 10:
				R = POW(rr,CST(1.0E+1))*-CST(5.975575286115054E+1)+POW(rr,CST(1.2E+1))*CST(6.49519052838441E+1);
				break;
			case 12:
				R = POW(rr,CST(1.2E+1))*CST(5.19615242270811);
				break;
			} 
			break;
#endif // L1 >= 12
#if L1 >= 13
    case 13:
				switch (n)
				{
				case 1:
					R = rr*CST(7.896313435467891E+1)-(r2*rr)*CST(1.610847940832376E+3)+(r2*r2*rr)*CST(1.093075388422608E+4)-(r2*r2*r2*rr)*CST(3.400678986203671E+4)+(r2*r2*r2*r2*rr)*CST(5.332882955634594E+4)-POW(rr,CST(1.1E+1))*CST(4.102217658185959E+4)+POW(rr,CST(1.3E+1))*CST(1.230665297454596E+4);
					break;
				case 3:
					R = (r2*rr)*-CST(4.602422688100487E+2)+(r2*r2*rr)*CST(4.858112837433815E+3)-(r2*r2*r2*rr)*CST(1.854915810656548E+4)+(r2*r2*r2*r2*rr)*CST(3.281774126553535E+4)-POW(rr,CST(1.1E+1))*CST(2.734811772125959E+4)+POW(rr,CST(1.3E+1))*CST(8.687049158513546E+3);
					break;
				case 5:
					R = (r2*r2*rr)*CST(8.832932431697845E+2)-(r2*r2*r2*rr)*CST(5.707433263555169E+3)+(r2*r2*r2*r2*rr)*CST(1.312709650617838E+4)-POW(rr,CST(1.1E+1))*CST(1.286970245704055E+4)+POW(rr,CST(1.3E+1))*CST(4.572131136059761E+3);
					break;
				case 7:
					R = (r2*r2*r2*rr)*-CST(7.60991101808846E+2)+(r2*r2*r2*r2*rr)*CST(3.088728589691222E+3)-POW(rr,CST(1.1E+1))*CST(4.064116565383971E+3)+POW(rr,CST(1.3E+1))*CST(1.741764242306352E+3);
					break;
				case 9:
					R = (r2*r2*r2*r2*rr)*CST(3.251293252306059E+2)-POW(rr,CST(1.1E+1))*CST(7.741174410264939E+2)+POW(rr,CST(1.3E+1))*CST(4.54373280601576E+2);
					break;
				case 11:
					R = POW(rr,CST(1.1E+1))*-CST(6.731456008902751E+1)+POW(rr,CST(1.3E+1))*CST(7.269972489634529E+1);
					break;
				case 13:
					R = POW(rr,CST(1.3E+1))*CST(5.385164807128604);
					break;
				} 
				break;
#endif // L1 >= 13
#if L1 >= 14
    case 14:
        switch (n)
        {
        case 0:
            R = (r2)*CST(6.939451623205096E+2)-(r2*r2)*CST(7.910974850460887E+3)+(r2*r2*r2)*CST(3.955487425231934E+4)-(r2*r2*r2*r2)*CST(1.010846786448956E+5)+POW(rr,CST(1.0E+1))*CST(1.378427436065674E+5)-POW(rr,CST(1.2E+1))*CST(9.542959172773361E+4)+POW(rr,CST(1.4E+1))*CST(2.63567443819046E+4)-CST(1.749441585683962E+1);
            break;
        case 2:
            R = (r2)*CST(2.775780649287626E+2)-(r2*r2)*CST(4.520557057410479E+3)+(r2*r2*r2)*CST(2.636991616821289E+4)-(r2*r2*r2*r2)*CST(7.351612992358208E+4)+POW(rr,CST(1.0E+1))*CST(1.060328796973228E+5)-POW(rr,CST(1.2E+1))*CST(7.634367338204384E+4)+POW(rr,CST(1.4E+1))*CST(2.170555419689417E+4);
            break;
        case 4:
            R = (r2*r2)*-CST(1.004568234980106E+3)+(r2*r2*r2)*CST(9.589060424804688E+3)-(r2*r2*r2*r2)*CST(3.393052150321007E+4)+POW(rr,CST(1.0E+1))*CST(5.655086917197704E+4)-POW(rr,CST(1.2E+1))*CST(4.490804316592216E+4)+POW(rr,CST(1.4E+1))*CST(1.370877107170224E+4);
            break;
        case 6:
            R = (r2*r2*r2)*CST(1.475240065354854E+3)-(r2*r2*r2*r2)*CST(9.04813906750083E+3)+POW(rr,CST(1.0E+1))*CST(1.99591302959919E+4)-POW(rr,CST(1.2E+1))*CST(1.8908649754107E+4)+POW(rr,CST(1.4E+1))*CST(6.527986224621534E+3);
            break;
        case 8:
            R = (r2*r2*r2*r2)*-CST(1.064486949119717E+3)+POW(rr,CST(1.0E+1))*CST(4.201922167569399E+3)-POW(rr,CST(1.2E+1))*CST(5.402471358314157E+3)+POW(rr,CST(1.4E+1))*CST(2.270603904217482E+3);
            break;
        case 10:
            R = POW(rr,CST(1.0E+1))*CST(4.001830635787919E+2)-POW(rr,CST(1.2E+1))*CST(9.395602362267673E+2)+POW(rr,CST(1.4E+1))*CST(5.44944937011227E+2);
            break;
        case 12:
            R = POW(rr,CST(1.2E+1))*-CST(7.516481889830902E+1)+POW(rr,CST(1.4E+1))*CST(8.073258326109499E+1);
            break;
        case 14:
            R = POW(rr,CST(1.4E+1))*CST(5.567764362829621);
            break;
        } 
		break;
#endif // L1 >= 14
#if L1 >= 15
			case 15:
				switch (n)
				{
				case 1:
					R = rr*-CST(1.022829477079213E+2)+(r2*rr)*CST(2.720726409032941E+3)-(r2*r2*rr)*CST(2.448653768128157E+4)+(r2*r2*r2*rr)*CST(1.042945123462677E+5)-(r2*r2*r2*r2*rr)*CST(2.370329826049805E+5)+POW(rr,CST(1.1E+1))*CST(2.953795629386902E+5)-POW(rr,CST(1.3E+1))*CST(1.903557183384895E+5)+POW(rr,CST(1.5E+1))*CST(4.958846444106102E+4);
					break;
				case 3:
					R = (r2*rr)*CST(7.773504025805742E+2)-(r2*r2*rr)*CST(1.088290563613176E+4)+(r2*r2*r2*rr)*CST(5.688791582524776E+4)-(r2*r2*r2*r2*rr)*CST(1.458664508337975E+5)+POW(rr,CST(1.1E+1))*CST(1.969197086257935E+5)-POW(rr,CST(1.3E+1))*CST(1.343687423563004E+5)+POW(rr,CST(1.5E+1))*CST(3.653886853551865E+4);
					break;
				case 5:
					R = (r2*r2*rr)*-CST(1.978710115659982E+3)+(r2*r2*r2*rr)*CST(1.750397410005331E+4)-(r2*r2*r2*r2*rr)*CST(5.834658033359051E+4)+POW(rr,CST(1.1E+1))*CST(9.266809817695618E+4)-POW(rr,CST(1.3E+1))*CST(7.072039071393013E+4)+POW(rr,CST(1.5E+1))*CST(2.08793534488678E+4);
					break;
				case 7:
					R = (r2*r2*r2*rr)*CST(2.333863213345408E+3)-(r2*r2*r2*r2*rr)*CST(1.372860713732243E+4)+POW(rr,CST(1.1E+1))*CST(2.926360995060205E+4)-POW(rr,CST(1.3E+1))*CST(2.694110122436285E+4)+POW(rr,CST(1.5E+1))*CST(9.077979760378599E+3);
					break;
				case 9:
					R = (r2*r2*r2*r2*rr)*-CST(1.445116540770978E+3)+POW(rr,CST(1.1E+1))*CST(5.57402094297111E+3)-POW(rr,CST(1.3E+1))*CST(7.028113362878561E+3)+POW(rr,CST(1.5E+1))*CST(2.90495352332294E+3);
					break;
				case 11:
					R = POW(rr,CST(1.1E+1))*CST(4.846974733015522E+2)-POW(rr,CST(1.3E+1))*CST(1.124498138058931E+3)+POW(rr,CST(1.5E+1))*CST(6.455452274046838E+2);
					break;
				case 13:
					R = POW(rr,CST(1.3E+1))*-CST(8.329615837475285E+1)+POW(rr,CST(1.5E+1))*CST(8.904072102135979E+1);
					break;
				case 15:
					R = POW(rr,CST(1.5E+1))*CST(5.744562646534177);
					break;
				} 
				break;
#endif // L1 >=15
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
#if L2 >= 8
			case 8:
				switch (m)
				{
				case -8:
					Y = sinp*POW(cost2-CST(1.0),CST(4.0))*CST(7.289266601746931E-1);
					break;
				case -7:
					Y = cost*sinp*POW(cost2*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*CST(2.915706640698772);
					break;
				case -6:
					Y = sinp*(cost2*CST(1.0135125E+6)-CST(6.75675E+4))*POW(cost2-CST(1.0),CST(3.0))*-CST(7.878532816224526E-6);
					break;
				case -5:
					Y = sinp*POW(cost2*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost*CST(6.75675E+4)-(cost2*cost)*CST(3.378375E+5))*-CST(5.105872826582925E-5);
					break;
				case -4:
					Y = sinp*POW(cost2-CST(1.0),CST(2.0))*(cost2*-CST(3.378375E+4)+(cost2*cost2)*CST(8.4459375E+4)+CST(1.299375E+3))*CST(3.681897256448963E-4);
					break;
				case -3:
					Y = sinp*POW(cost2*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*(cost*CST(1.299375E+3)-(cost*cost2)*CST(1.126125E+4)+(cost*cost2*cost2)*CST(1.6891875E+4))*CST(2.851985351334463E-3);
					break;
				case -2:
					Y = sinp*(cost2-CST(1.0))*(cost2*CST(6.496875E+2)-(cost2*cost2)*CST(2.8153125E+3)+(cost2*cost2*cost2)*CST(2.8153125E+3)-CST(1.96875E+1))*-CST(2.316963852365461E-2);
					break;
				case -1:
					Y = sinp*sqrt(cost2*-CST(1.0)+CST(1.0))*(cost*CST(1.96875E+1)-(cost*cost2)*CST(2.165625E+2)+(cost*cost2*cost2)*CST(5.630625E+2)-(cost*cost2*cost2*cost2)*CST(4.021875E+2))*-CST(1.938511038201796E-1);;
					break;
				case 0:
					Y = cost2*-CST(1.144933081936324E+1)+(cost2*cost2)*CST(6.297131950652692E+1)-(cost2*cost2*cost2)*CST(1.091502871445846E+2)+(cost2*cost2*cost2*cost2)*CST(5.847336811327841E+1)+CST(3.180369672045344E-1);
					break;
				case 1:
					Y = cosp*sqrt(cost2*-CST(1.0)+CST(1.0))*(cost*CST(1.96875E+1)-(cost*cost2)*CST(2.165625E+2)+(cost*cost2*cost2)*CST(5.630625E+2)-(cost*cost2*cost2*cost2)*CST(4.021875E+2))*-CST(1.938511038201796E-1);;
					break;
				case 2:
					Y = cosp*(cost2-CST(1.0))*(cost2*CST(6.496875E+2)-(cost2*cost2)*CST(2.8153125E+3)+(cost2*cost2*cost2)*CST(2.8153125E+3)-CST(1.96875E+1))*-CST(2.316963852365461E-2);
					break;
				case 3:
					Y = cosp*POW(cost2*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*(cost*CST(1.299375E+3)-(cost*cost2)*CST(1.126125E+4)+(cost*cost2*cost2)*CST(1.6891875E+4))*CST(2.851985351334463E-3);
					break;
				case 4:
					Y = cosp*POW(cost2-CST(1.0),CST(2.0))*(cost2*-CST(3.378375E+4)+(cost2*cost2)*CST(8.4459375E+4)+CST(1.299375E+3))*CST(3.681897256448963E-4);
					break;
				case 5:
					Y = cosp*POW(cost2*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost*CST(6.75675E+4)-(cost*cost2)*CST(3.378375E+5))*-CST(5.105872826582925E-5);
					break;
				case 6:
					Y = cosp*(cost2*CST(1.0135125E+6)-CST(6.75675E+4))*POW(cost2-CST(1.0),CST(3.0))*-CST(7.878532816224526E-6);
					break;
				case 7:
					Y = cosp*cost*POW(cost2*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*CST(2.915706640698772);
					break;
				case 8:
					Y = cosp*POW(cost2-CST(1.0),CST(4.0))*CST(7.289266601746931E-1);
					break;
				}
				break;
#endif // L2 >= 8
#if L2 >= 9
			case 9:
				switch (m)
				{
				case -9:
					Y = sinp*POW(cost2*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*CST(7.489009518540115E-1);
					break;
				case -8:
					Y = cost*sinp*POW(cost2-CST(1.0),CST(4.0))*CST(3.17731764895143);
					break;
				case -7:
					Y = sinp*POW(cost2*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*(cost2*CST(1.72297125E+7)-CST(1.0135125E+6))*CST(5.376406125665728E-7);
					break;
				case -6:
					Y = sinp*(cost*CST(1.0135125E+6)-(cost*cost2)*CST(5.7432375E+6))*POW(cost2-CST(1.0),CST(3.0))*CST(3.724883428715686E-6);
					break;
				case -5:
					Y = sinp*POW(cost2*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost2*-CST(5.0675625E+5)+(cost2*cost2)*CST(1.435809375E+6)+CST(1.6891875E+4))*CST(2.885282297193648E-5);
					break;
				case -4:
					Y = sinp*POW(cost2-CST(1.0),CST(2.0))*(cost*CST(1.6891875E+4)-(cost*cost2)*CST(1.6891875E+5)+(cost*cost2*cost2)*CST(2.87161875E+5))*CST(2.414000363328839E-4);
					break;
				case -3:
					Y = sinp*POW(cost2*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*((cost2)*CST(8.4459375E+3)-(cost2*cost2)*CST(4.22296875E+4)+(cost2*cost2*cost2)*CST(4.78603125E+4)-CST(2.165625E+2))*CST(2.131987394015766E-3);
					break;
				case -2:
					Y = sinp*(cost2-CST(1.0))*(cost*CST(2.165625E+2)-(cost*cost2)*CST(2.8153125E+3)+(cost*cost2*cost2)*CST(8.4459375E+3)-(cost*cost2*cost2*cost2)*CST(6.8371875E+3))*CST(1.953998722751749E-2);
					break;
				case -1:
					Y = sinp*sqrt(cost2*-CST(1.0)+CST(1.0))*(cost2*-CST(1.0828125E+2)+(cost2*cost2)*CST(7.03828125E+2)-(cost2*cost2*cost2)*CST(1.40765625E+3)+(cost2*cost2*cost2*cost2)*CST(8.546484375E+2)+CST(2.4609375))*CST(1.833013280775049E-1);
					break;
				case 0:
					Y = cost*CST(3.026024588281871)-(cost*cost2)*CST(4.438169396144804E+1)+(cost*cost2*cost2)*CST(1.730886064497754E+2)-(cost*cost2*cost2*cost2)*CST(2.472694377852604E+2)+(cost*cost2*cost2*cost2*cost2)*CST(1.167661233986728E+2);
					break;
				case 1:
					Y = cosp*sqrt(cost2*-CST(1.0)+CST(1.0))*(cost2*-CST(1.0828125E+2)+(cost2*cost2)*CST(7.03828125E+2)-(cost2*cost2*cost2)*CST(1.40765625E+3)+(cost2*cost2*cost2*cost2)*CST(8.546484375E+2)+CST(2.4609375))*CST(1.833013280775049E-1);
					break;
				case 2:
					Y = cosp*(cost2-CST(1.0))*(cost*CST(2.165625E+2)-(cost*cost2)*CST(2.8153125E+3)+(cost*cost2*cost2)*CST(8.4459375E+3)-(cost*cost2*cost2*cost2)*CST(6.8371875E+3))*CST(1.953998722751749E-2);
					break;
				case 3:
					Y = cosp*POW(cost2*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*(cost2*CST(8.4459375E+3)-(cost2*cost2)*CST(4.22296875E+4)+(cost2*cost2*cost2)*CST(4.78603125E+4)-CST(2.165625E+2))*CST(2.131987394015766E-3);
					break;
				case 4:
					Y = cosp*POW(cost2-CST(1.0),CST(2.0))*(cost*CST(1.6891875E+4)-(cost*cost2)*CST(1.6891875E+5)+(cost*cost2*cost2)*CST(2.87161875E+5))*CST(2.414000363328839E-4);
					break;
				case 5:
					Y = cosp*POW(cost2*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost2*-CST(5.0675625E+5)+(cost2*cost2)*CST(1.435809375E+6)+CST(1.6891875E+4))*CST(2.885282297193648E-5);
					break;
				case 6:
					Y = cosp*(cost*CST(1.0135125E+6)-(cost*cost2)*CST(5.7432375E+6))*POW(cost2-CST(1.0),CST(3.0))*CST(3.724883428715686E-6);
					break;
				case 7:
					Y = cosp*POW(cost2*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*(cost2*CST(1.72297125E+7)-CST(1.0135125E+6))*CST(5.376406125665728E-7);
					break;
				case 8:
					Y = cosp*cost*POW(cost2-CST(1.0),CST(4.0))*CST(3.17731764895143);
					break;
				case 9:
					Y = cosp*POW(cost2*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*CST(7.489009518540115E-1);
					break;
				}
				break;
#endif // L2 >= 9
#if L2 >= 10
			case 10:
				switch (m)
				{
				case -10:
					Y = sinp*POW(cost*cost-CST(1.0),CST(5.0))*-CST(7.673951182223391E-1);
					break;
				case -9:
					Y = cost*sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*CST(3.431895299894677);
					break;
				case -8:
					Y = sinp*((cost*cost)*CST(3.273645375E+8)-CST(1.72297125E+7))*POW(cost*cost-CST(1.0),CST(4.0))*CST(3.231202683857352E-8);
					break;
				case -7:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*(cost*CST(1.72297125E+7)-(cost*cost*cost)*CST(1.091215125E+8))*-CST(2.374439349284684E-7);
					break;
				case -6:
					Y = sinp*POW(cost*cost-CST(1.0),CST(3.0))*((cost*cost)*-CST(8.61485625E+6)+(cost*cost*cost*cost)*CST(2.7280378125E+7)+CST(2.53378125E+5))*-CST(1.958012847746993E-6);
					break;
				case -5:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost*CST(2.53378125E+5)-(cost*cost*cost)*CST(2.87161875E+6)+(cost*cost*cost*cost*cost)*CST(5.456075625E+6))*CST(1.751299931351813E-5);
					break;
				case -4:
					Y = sinp*POW(cost*cost-CST(1.0),CST(2.0))*((cost*cost)*CST(1.266890625E+5)-(cost*cost*cost*cost)*CST(7.179046875E+5)+(cost*cost*cost*cost*cost*cost)*CST(9.093459375E+5)-CST(2.8153125E+3))*CST(1.661428994750302E-4);
					break;
				case -3:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*(cost*CST(2.8153125E+3)-(cost*cost*cost)*CST(4.22296875E+4)+(cost*cost*cost*cost*cost)*CST(1.435809375E+5)-(cost*cost*cost*cost*cost*cost*cost)*CST(1.299065625E+5))*-CST(1.644730792108362E-3);
					break;
				case -2:
					Y = sinp*(cost*cost-CST(1.0))*((cost*cost)*-CST(1.40765625E+3)+(cost*cost*cost*cost)*CST(1.0557421875E+4)-(cost*cost*cost*cost*cost*cost)*CST(2.393015625E+4)+(cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.62383203125E+4)+CST(2.70703125E+1))*-CST(1.67730288071084E-2);
					break;
				case -1:
					Y = sinp*sqrt((cost*cost)*-CST(1.0)+CST(1.0))*(cost*CST(2.70703125E+1)-(cost*cost*cost)*CST(4.6921875E+2)+(cost*cost*cost*cost*cost)*CST(2.111484375E+3)-(cost*cost*cost*cost*cost*cost*cost)*CST(3.41859375E+3)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.8042578125E+3))*CST(1.743104285446861E-1);
					break;
				case 0:
					Y = (cost*cost)*CST(1.749717715557199E+1)-(cost*cost*cost*cost)*CST(1.516422020150349E+2)+(cost*cost*cost*cost*cost*cost)*CST(4.549266060441732E+2)-(cost*cost*cost*cost*cost*cost*cost*cost)*CST(5.524108787681907E+2)+POW(cost,CST(1.0E+1))*CST(2.332401488134637E+2)-CST(3.181304937370442E-1);
					break;
				case 1:
					Y = cosp*sqrt((cost*cost)*-CST(1.0)+CST(1.0))*(cost*CST(2.70703125E+1)-(cost*cost*cost)*CST(4.6921875E+2)+(cost*cost*cost*cost*cost)*CST(2.111484375E+3)-(cost*cost*cost*cost*cost*cost*cost)*CST(3.41859375E+3)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.8042578125E+3))*CST(1.743104285446861E-1);
					break;
				case 2:
					Y = cosp*(cost*cost-CST(1.0))*((cost*cost)*-CST(1.40765625E+3)+(cost*cost*cost*cost)*CST(1.0557421875E+4)-(cost*cost*cost*cost*cost*cost)*CST(2.393015625E+4)+(cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.62383203125E+4)+CST(2.70703125E+1))*-CST(1.67730288071084E-2);
					break;
				case 3:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*(cost*CST(2.8153125E+3)-(cost*cost*cost)*CST(4.22296875E+4)+(cost*cost*cost*cost*cost)*CST(1.435809375E+5)-(cost*cost*cost*cost*cost*cost*cost)*CST(1.299065625E+5))*-CST(1.644730792108362E-3);
					break;
				case 4:
					Y = cosp*POW(cost*cost-CST(1.0),CST(2.0))*((cost*cost)*CST(1.266890625E+5)-(cost*cost*cost*cost)*CST(7.179046875E+5)+(cost*cost*cost*cost*cost*cost)*CST(9.093459375E+5)-CST(2.8153125E+3))*CST(1.661428994750302E-4);
					break;
				case 5:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost*CST(2.53378125E+5)-(cost*cost*cost)*CST(2.87161875E+6)+(cost*cost*cost*cost*cost)*CST(5.456075625E+6))*CST(1.751299931351813E-5);
					break;
				case 6:
					Y = cosp*POW(cost*cost-CST(1.0),CST(3.0))*((cost*cost)*-CST(8.61485625E+6)+(cost*cost*cost*cost)*CST(2.7280378125E+7)+CST(2.53378125E+5))*-CST(1.958012847746993E-6);
					break;
				case 7:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*(cost*CST(1.72297125E+7)-(cost*cost*cost)*CST(1.091215125E+8))*-CST(2.374439349284684E-7);
					break;
				case 8:
					Y = cosp*((cost*cost)*CST(3.273645375E+8)-CST(1.72297125E+7))*POW(cost*cost-CST(1.0),CST(4.0))*CST(3.231202683857352E-8);
					break;
				case 9:
					Y = cosp*cost*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*CST(3.431895299894677);
					break;
				case 10:
					Y = cosp*POW(cost*cost-CST(1.0),CST(5.0))*-CST(7.673951182223391E-1);
					break;
				}
				break;
#endif // L2 >= 10
#if L2 >= 11
			case 11:
				switch (m)
				{
				case -11:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(1.1E+1)/CST(2.0))*CST(7.846421057874977E-1);
					break;
				case -10:
					Y = cost*sinp*POW(cost*cost-CST(1.0),CST(5.0))*-CST(3.68029769880377);
					break;
				case -9:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*((cost*cost)*CST(6.8746552875E+9)-CST(3.273645375E+8))*CST(1.734709165873547E-9);
					break;
				case -8:
					Y = sinp*POW(cost*cost-CST(1.0),CST(4.0))*(cost*CST(3.273645375E+8)-(cost*cost*cost)*CST(2.2915517625E+9))*-CST(1.343699941990114E-8);
					break;
				case -7:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*((cost*cost)*-CST(1.6368226875E+8)+(cost*cost*cost*cost)*CST(5.72887940625E+8)+CST(4.307428125E+6))*CST(1.171410451514688E-7);
					break;
				case -6:
					Y = sinp*POW(cost*cost-CST(1.0),CST(3.0))*(cost*CST(4.307428125E+6)-(cost*cost*cost)*CST(5.456075625E+7)+(cost*cost*cost*cost*cost)*CST(1.14577588125E+8))*-CST(1.111297530512201E-6);
					break;
				case -5:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*((cost*cost)*CST(2.1537140625E+6)-(cost*cost*cost*cost)*CST(1.36401890625E+7)+(cost*cost*cost*cost*cost*cost)*CST(1.90962646875E+7)-CST(4.22296875E+4))*CST(1.122355489741045E-5);
					break;
				case -4:
					Y = sinp*POW(cost*cost-CST(1.0),CST(2.0))*(cost*CST(4.22296875E+4)-(cost*cost*cost)*CST(7.179046875E+5)+(cost*cost*cost*cost*cost)*CST(2.7280378125E+6)-(cost*cost*cost*cost*cost*cost*cost)*CST(2.7280378125E+6))*-CST(1.187789403385153E-4);
					break;
				case -3:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*((cost*cost)*-CST(2.111484375E+4)+(cost*cost*cost*cost)*CST(1.79476171875E+5)-(cost*cost*cost*cost*cost*cost)*CST(4.5467296875E+5)+(cost*cost*cost*cost*cost*cost*cost*cost)*CST(3.410047265625E+5)+CST(3.519140625E+2))*CST(1.301158099600741E-3);
					break;
				case -2:
					Y = sinp*(cost*cost-CST(1.0))*(cost*CST(3.519140625E+2)-(cost*cost*cost)*CST(7.03828125E+3)+(cost*cost*cost*cost*cost)*CST(3.5895234375E+4)-(cost*cost*cost*cost*cost*cost*cost)*CST(6.495328125E+4)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(3.78894140625E+4))*-CST(1.46054634441839E-2);
					break;
				case -1:
					Y = sinp*sqrt((cost*cost)*-CST(1.0)+CST(1.0))*((cost*cost)*CST(1.7595703125E+2)-(cost*cost*cost*cost)*CST(1.7595703125E+3)+(cost*cost*cost*cost*cost*cost)*CST(5.9825390625E+3)-(cost*cost*cost*cost*cost*cost*cost*cost)*CST(8.11916015625E+3)+POW(cost,CST(1.0E+1))*CST(3.78894140625E+3)-CST(2.70703125))*CST(1.665279049125274E-1);
					break;
				case 0:
					Y = cost*-CST(3.662285987506039)+(cost*cost*cost)*CST(7.934952972922474E+1)-(cost*cost*cost*cost*cost)*CST(4.760971783753484E+2)+(cost*cost*cost*cost*cost*cost*cost)*CST(1.156236004628241E+3)-(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.220471338216215E+3)+POW(cost,CST(1.1E+1))*CST(4.65998147319071E+2);
					break;
				case 1:
					Y = cosp*sqrt((cost*cost)*-CST(1.0)+CST(1.0))*((cost*cost)*CST(1.7595703125E+2)-(cost*cost*cost*cost)*CST(1.7595703125E+3)+(cost*cost*cost*cost*cost*cost)*CST(5.9825390625E+3)-(cost*cost*cost*cost*cost*cost*cost*cost)*CST(8.11916015625E+3)+POW(cost,CST(1.0E+1))*CST(3.78894140625E+3)-CST(2.70703125))*CST(1.665279049125274E-1);
					break;
				case 2:
					Y = cosp*(cost*cost-CST(1.0))*(cost*CST(3.519140625E+2)-(cost*cost*cost)*CST(7.03828125E+3)+(cost*cost*cost*cost*cost)*CST(3.5895234375E+4)-(cost*cost*cost*cost*cost*cost*cost)*CST(6.495328125E+4)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(3.78894140625E+4))*-CST(1.46054634441839E-2);
					break;
				case 3:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*((cost*cost)*-CST(2.111484375E+4)+(cost*cost*cost*cost)*CST(1.79476171875E+5)-(cost*cost*cost*cost*cost*cost)*CST(4.5467296875E+5)+(cost*cost*cost*cost*cost*cost*cost*cost)*CST(3.410047265625E+5)+CST(3.519140625E+2))*CST(1.301158099600741E-3);
					break;
				case 4:
					Y = cosp*POW(cost*cost-CST(1.0),CST(2.0))*(cost*CST(4.22296875E+4)-(cost*cost*cost)*CST(7.179046875E+5)+(cost*cost*cost*cost*cost)*CST(2.7280378125E+6)-(cost*cost*cost*cost*cost*cost*cost)*CST(2.7280378125E+6))*-CST(1.187789403385153E-4);
					break;
				case 5:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*((cost*cost)*CST(2.1537140625E+6)-(cost*cost*cost*cost)*CST(1.36401890625E+7)+(cost*cost*cost*cost*cost*cost)*CST(1.90962646875E+7)-CST(4.22296875E+4))*CST(1.122355489741045E-5);
					break;
				case 6:
					Y = cosp*POW(cost*cost-CST(1.0),CST(3.0))*(cost*CST(4.307428125E+6)-(cost*cost*cost)*CST(5.456075625E+7)+(cost*cost*cost*cost*cost)*CST(1.14577588125E+8))*-CST(1.111297530512201E-6);
					break;
				case 7:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*((cost*cost)*-CST(1.6368226875E+8)+(cost*cost*cost*cost)*CST(5.72887940625E+8)+CST(4.307428125E+6))*CST(1.171410451514688E-7);
					break;
				case 8:
					Y = cosp*POW(cost*cost-CST(1.0),CST(4.0))*(cost*CST(3.273645375E+8)-(cost*cost*cost)*CST(2.2915517625E+9))*-CST(1.343699941990114E-8);
					break;
				case 9:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*((cost*cost)*CST(6.8746552875E+9)-CST(3.273645375E+8))*CST(1.734709165873547E-9);
					break;
				case 10:
					Y = cosp*cost*POW(cost*cost-CST(1.0),CST(5.0))*-CST(3.68029769880377);
					break;
				case 11:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(1.1E+1)/CST(2.0))*CST(7.846421057874977E-1);
					break;
				}
				break;
#endif // L2 >= 11
#if L2 >=  12
			case 12:
				switch (m)
				{
				case -12:
					Y = sinp*POW(cost*cost-CST(1.0),CST(6.0))*CST(8.00821995784645E-1);
					break;
				case -11:
					Y = cost*sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(1.1E+1)/CST(2.0))*CST(3.923210528933851);
					break;
				case -10:
					Y = sinp*((cost*cost)*CST(1.581170716125E+11)-CST(6.8746552875E+9))*POW(cost*cost-CST(1.0),CST(5.0))*-CST(8.414179483959553E-11);
					break;
				case -9:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*(cost*CST(6.8746552875E+9)-(cost*cost*cost)*CST(5.27056905375E+10))*-CST(6.83571172712202E-10);
					break;
				case -8:
					Y = sinp*POW(cost*cost-CST(1.0),CST(4.0))*((cost*cost)*-CST(3.43732764375E+9)+(cost*cost*cost*cost)*CST(1.3176422634375E+10)+CST(8.1841134375E+7))*CST(6.265033283689913E-9);
					break;
				case -7:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*(cost*CST(8.1841134375E+7)-(cost*cost*cost)*CST(1.14577588125E+9)+(cost*cost*cost*cost*cost)*CST(2.635284526875E+9))*CST(6.26503328367365E-8);
					break;
				case -6:
					Y = sinp*POW(cost*cost-CST(1.0),CST(3.0))*((cost*cost)*CST(4.09205671875E+7)-(cost*cost*cost*cost)*CST(2.864439703125E+8)+(cost*cost*cost*cost*cost*cost)*CST(4.392140878125E+8)-CST(7.179046875E+5))*-CST(6.689225062143228E-7);
					break;
				case -5:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost*CST(7.179046875E+5)-(cost*cost*cost)*CST(1.36401890625E+7)+(cost*cost*cost*cost*cost)*CST(5.72887940625E+7)-(cost*cost*cost*cost*cost*cost*cost)*CST(6.27448696875E+7))*-CST(7.50863650966771E-6);
					break;
				case -4:
					Y = sinp*POW(cost*cost-CST(1.0),CST(2.0))*((cost*cost)*-CST(3.5895234375E+5)+(cost*cost*cost*cost)*CST(3.410047265625E+6)-(cost*cost*cost*cost*cost*cost)*CST(9.54813234375E+6)+(cost*cost*cost*cost*cost*cost*cost*cost)*CST(7.8431087109375E+6)+CST(5.2787109375E+3))*CST(8.756499656747962E-5);
					break;
				case -3:
					Y = sinp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*(cost*CST(5.2787109375E+3)-(cost*cost*cost)*CST(1.1965078125E+5)+(cost*cost*cost*cost*cost)*CST(6.82009453125E+5)-(cost*cost*cost*cost*cost*cost*cost)*CST(1.36401890625E+6)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(8.714565234375E+5))*CST(1.050779958809755E-3);
					break;
				case -2:
					Y = sinp*(cost*cost-CST(1.0))*((cost*cost)*CST(2.63935546875E+3)-(cost*cost*cost*cost)*CST(2.99126953125E+4)+(cost*cost*cost*cost*cost*cost)*CST(1.136682421875E+5)-(cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.7050236328125E+5)+POW(cost,CST(1.0E+1))*CST(8.714565234375E+4)-CST(3.519140625E+1))*-CST(1.286937365514973E-2);
					break;
				case -1:
					Y = sinp*sqrt((cost*cost)*-CST(1.0)+CST(1.0))*(cost*CST(3.519140625E+1)-(cost*cost*cost)*CST(8.7978515625E+2)+(cost*cost*cost*cost*cost)*CST(5.9825390625E+3)-(cost*cost*cost*cost*cost*cost*cost)*CST(1.62383203125E+4)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.894470703125E+4)-POW(cost,CST(1.1E+1))*CST(7.92233203125E+3))*-CST(1.597047270888652E-1);
					break;
				case 0:
					Y = (cost*cost)*-CST(2.481828104582382E+1)+(cost*cost*cost*cost)*CST(3.102285130722448E+2)-(cost*cost*cost*cost*cost*cost)*CST(1.40636925926432E+3)+(cost*cost*cost*cost*cost*cost*cost*cost)*CST(2.862965992070735E+3)-POW(cost,CST(1.0E+1))*CST(2.672101592600346E+3)+POW(cost,CST(1.2E+1))*CST(9.311869186330587E+2)+CST(3.181830903313312E-1);
					break;
				case 1:
					Y = cosp*sqrt((cost*cost)*-CST(1.0)+CST(1.0))*(cost*CST(3.519140625E+1)-(cost*cost*cost)*CST(8.7978515625E+2)+(cost*cost*cost*cost*cost)*CST(5.9825390625E+3)-(cost*cost*cost*cost*cost*cost*cost)*CST(1.62383203125E+4)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.894470703125E+4)-POW(cost,CST(1.1E+1))*CST(7.92233203125E+3))*-CST(1.597047270888652E-1);
					break;
				case 2:
					Y = cosp*(cost*cost-CST(1.0))*((cost*cost)*CST(2.63935546875E+3)-(cost*cost*cost*cost)*CST(2.99126953125E+4)+(cost*cost*cost*cost*cost*cost)*CST(1.136682421875E+5)-(cost*cost*cost*cost*cost*cost*cost*cost)*CST(1.7050236328125E+5)+POW(cost,CST(1.0E+1))*CST(8.714565234375E+4)-CST(3.519140625E+1))*-CST(1.286937365514973E-2);
					break;
				case 3:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(3.0)/CST(2.0))*(cost*CST(5.2787109375E+3)-(cost*cost*cost)*CST(1.1965078125E+5)+(cost*cost*cost*cost*cost)*CST(6.82009453125E+5)-(cost*cost*cost*cost*cost*cost*cost)*CST(1.36401890625E+6)+(cost*cost*cost*cost*cost*cost*cost*cost*cost)*CST(8.714565234375E+5))*CST(1.050779958809755E-3);
					break;
				case 4:
					Y = cosp*POW(cost*cost-CST(1.0),CST(2.0))*((cost*cost)*-CST(3.5895234375E+5)+(cost*cost*cost*cost)*CST(3.410047265625E+6)-(cost*cost*cost*cost*cost*cost)*CST(9.54813234375E+6)+(cost*cost*cost*cost*cost*cost*cost*cost)*CST(7.8431087109375E+6)+CST(5.2787109375E+3))*CST(8.756499656747962E-5);
					break;
				case 5:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(5.0)/CST(2.0))*(cost*CST(7.179046875E+5)-(cost*cost*cost)*CST(1.36401890625E+7)+(cost*cost*cost*cost*cost)*CST(5.72887940625E+7)-(cost*cost*cost*cost*cost*cost*cost)*CST(6.27448696875E+7))*-CST(7.50863650966771E-6);
					break;
				case 6:
					Y = cosp*POW(cost*cost-CST(1.0),CST(3.0))*((cost*cost)*CST(4.09205671875E+7)-(cost*cost*cost*cost)*CST(2.864439703125E+8)+(cost*cost*cost*cost*cost*cost)*CST(4.392140878125E+8)-CST(7.179046875E+5))*-CST(6.689225062143228E-7);
					break;
				case 7:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(7.0)/CST(2.0))*(cost*CST(8.1841134375E+7)-(cost*cost*cost)*CST(1.14577588125E+9)+(cost*cost*cost*cost*cost)*CST(2.635284526875E+9))*CST(6.26503328367365E-8);
					break;
				case 8:
					Y = cosp*POW(cost*cost-CST(1.0),CST(4.0))*((cost*cost)*-CST(3.43732764375E+9)+(cost*cost*cost*cost)*CST(1.3176422634375E+10)+CST(8.1841134375E+7))*CST(6.265033283689913E-9);
					break;
				case 9:
					Y = cosp*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(9.0)/CST(2.0))*(cost*CST(6.8746552875E+9)-(cost*cost*cost)*CST(5.27056905375E+10))*-CST(6.83571172712202E-10);
					break;
				case 10:
					Y = cosp*((cost*cost)*CST(1.581170716125E+11)-CST(6.8746552875E+9))*POW(cost*cost-CST(1.0),CST(5.0))*-CST(8.414179483959553E-11);
					break;
				case 11:
					Y = cosp*cost*POW((cost*cost)*-CST(1.0)+CST(1.0),CST(1.1E+1)/CST(2.0))*CST(3.923210528933851);
					break;
				case 12:
					Y = cosp*POW(cost*cost-CST(1.0),CST(6.0))*CST(8.00821995784645E-1);
					break;
				}
				break;
#endif // L2 >= 12
		}

		return R * Y;
	}

	template<typename PrecisionType>
	__device__ PrecisionType atomicAddPrecision(PrecisionType *addr, PrecisionType val)
	{
		return atomicAdd(addr, val);
	}

	template<typename PrecisionType>
	__device__ PrecisionType safe_divide(PrecisionType numerator, PrecisionType denominator) 
	{
		if (denominator == CST(0.0)) {
			return CST(0.0); // return zero if denominator is zero
		} else {
			return numerator / denominator;
		}
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
								   const unsigned int loopStep,
								   MultidimArrayCuda<PrecisionType> &mP,
								   MultidimArrayCuda<PrecisionType> &mW,
								   const PrecisionType weight)
	{
		int i = static_cast<int>(CUDA_ROUND(pos_y));
		int j = static_cast<int>(CUDA_ROUND(pos_x));
		if (!IS_OUTSIDE2D(mP, i, j)) {
			PrecisionType m = 1. / loopStep;
			PrecisionType a = m * ABSC(static_cast<PrecisionType>(i) - pos_y);
			PrecisionType b = m * ABSC(static_cast<PrecisionType>(j) - pos_x);
			PrecisionType gw = 1. - a - b + a * b;
			atomicAddPrecision(&A2D_ELEM(mP, i, j), weight * gw);
			atomicAddPrecision(&A2D_ELEM(mW, i, j), gw * gw);
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
							  const PrecisionType loopStep,
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

	auto r_x = CST(j) + gx;
	auto r_y = CST(i) + gy;
	auto r_z = CST(k) + gz;

	auto pos_x = r0 * r_x + r1 * r_y + r2 * r_z;
	auto pos_y = r3 * r_x + r4 * r_y + r5 * r_z;
	device::splattingAtPos(pos_x, pos_y, loopStep, mP, mW, weight);
}

/*
 * The second beast
 */
template<typename PrecisionType, bool usesZernike>
__global__ void backwardKernel(MultidimArrayCuda<PrecisionType> cudaMV,
							   const MultidimArrayCuda<PrecisionType> cudaMId,
							   const MultidimArrayCuda<PrecisionType> cudaMIws,
							   const MultidimArrayCuda<PrecisionType> cudaMId_small,
							   const MultidimArrayCuda<PrecisionType> cudaMIws_small,
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
							   const PrecisionType *cudaR,
							   MultidimArrayCuda<PrecisionType> cudaReg,
							   double lmr)
{
	int cubeX = threadIdx.x + blockIdx.x * blockDim.x;
	int cubeY = threadIdx.y + blockIdx.y * blockDim.y;
	int cubeZ = threadIdx.z + blockIdx.z * blockDim.z;
	int k = STARTINGZ(cudaMV) + cubeZ;
	int i = STARTINGY(cudaMV) + cubeY;
	int j = STARTINGX(cudaMV) + cubeX;
	PrecisionType scale_factor = CST(XSIZE(cudaMId_small)) / CST(XSIZE(cudaMId));
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

		auto r_x = CST(j) + gx;
		auto r_y = CST(i) + gy;
		auto r_z = CST(k) + gz;

		auto pos_x = cudaR[0] * r_x + cudaR[1] * r_y + cudaR[2] * r_z;
		auto pos_y = cudaR[3] * r_x + cudaR[4] * r_y + cudaR[5] * r_z;
		PrecisionType voxel = device::interpolatedElement2DCuda(pos_x, pos_y, cudaMId);
		PrecisionType weight = device::interpolatedElement2DCuda(pos_x, pos_y, cudaMIws);
		// PrecisionType voxel_small = device::interpolatedElement2DCuda(scale_factor * pos_x, scale_factor * pos_y, cudaMId_small);
		// PrecisionType weight_small = device::interpolatedElement2DCuda(scale_factor * pos_x, scale_factor * pos_y, cudaMIws_small);
		
		A3D_ELEM(cudaMV, k, i, j) += (voxel / (weight + CST(1e-5))) + A3D_ELEM(cudaReg, k, i, j);

	}
}

template<typename PrecisionType>
__global__ void computeStdDevParams(MultidimArrayCuda<PrecisionType> cudaMV, PrecisionType *elems,
							  PrecisionType *avg, PrecisionType *sumSqrNorm, const MultidimArrayCuda<int> VRecMaskB)
{
	// Remove negative values
	int cubeX = threadIdx.x + blockIdx.x * blockDim.x;
	int cubeY = threadIdx.y + blockIdx.y * blockDim.y;
	int cubeZ = threadIdx.z + blockIdx.z * blockDim.z;
	int k = STARTINGZ(cudaMV) + cubeZ;
	int i = STARTINGY(cudaMV) + cubeY;
	int j = STARTINGX(cudaMV) + cubeX;
	if (A3D_ELEM(VRecMaskB, k, i, j) != 0) {
		device::atomicAddPrecision(elems, CST(1.0));
		device::atomicAddPrecision(avg, A3D_ELEM(cudaMV, k, i, j));
		device::atomicAddPrecision(sumSqrNorm, A3D_ELEM(cudaMV, k, i, j));
	}
}

template<typename PrecisionType>
__global__ void computeStdDev(PrecisionType *elems, PrecisionType *avg, PrecisionType *sumSqrNorm,
	                          PrecisionType *stddev)
{
	if (*elems != 0) {
		*avg /= *elems;
		*sumSqrNorm /= *elems;
	}
	*stddev = sqrtf(fabsf(*sumSqrNorm - (*avg * *avg)));
}

template<typename PrecisionType>
__global__ void softThreshold(MultidimArrayCuda<PrecisionType> cudaMV, PrecisionType *stddev, double thr, const MultidimArrayCuda<int> VRecMaskB)
{
	// Remove negative values
	int cubeX = threadIdx.x + blockIdx.x * blockDim.x;
	int cubeY = threadIdx.y + blockIdx.y * blockDim.y;
	int cubeZ = threadIdx.z + blockIdx.z * blockDim.z;
	int k = STARTINGZ(cudaMV) + cubeZ;
	int i = STARTINGY(cudaMV) + cubeY;
	int j = STARTINGX(cudaMV) + cubeX;
	PrecisionType thr_neg = *stddev;
	PrecisionType thr_pos = CST(thr) * thr_neg;
	if (A3D_ELEM(VRecMaskB, k, i, j) != 0) {
		if (A3D_ELEM(cudaMV, k, i, j)  > thr_pos)
			A3D_ELEM(cudaMV, k, i, j) = A3D_ELEM(cudaMV, k, i, j) - thr_pos;
		else if (A3D_ELEM(cudaMV, k, i, j)  < -thr_pos)
			A3D_ELEM(cudaMV, k, i, j) = A3D_ELEM(cudaMV, k, i, j) + thr_pos;
		else
			A3D_ELEM(cudaMV, k, i, j) = 0.0;
		// if (A3D_ELEM(cudaMV, k, i, j)  < -thr_neg)
		// 	A3D_ELEM(cudaMV, k, i, j) = A3D_ELEM(cudaMV, k, i, j) + thr_neg;
	}
}

template<typename PrecisionType>
__global__ void computeTV(MultidimArrayCuda<PrecisionType> cudaMV, MultidimArrayCuda<PrecisionType> cudaDx,
                          MultidimArrayCuda<PrecisionType> cudaDy, MultidimArrayCuda<PrecisionType> cudaDz,  MultidimArrayCuda<PrecisionType> cudaDl1,
						  const MultidimArrayCuda<int> VRecMaskB, double lambda, double ltv, double ltk, double ll1, double lst) {
	int cubeX = threadIdx.x + blockIdx.x * blockDim.x;
	int cubeY = threadIdx.y + blockIdx.y * blockDim.y;
	int cubeZ = threadIdx.z + blockIdx.z * blockDim.z;
	int k = STARTINGZ(cudaMV) + cubeZ;
	int i = STARTINGY(cudaMV) + cubeY;
	int j = STARTINGX(cudaMV) + cubeX;

    PrecisionType grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;

    // Compute gradients
	if (A3D_ELEM(VRecMaskB, k, i, j) != 0) {
		if (j < FINISHINGX(cudaMV) && j > STARTINGX(cudaMV)) grad_x = CST(0.5) * A3D_ELEM(cudaMV, k, i, j + 1) - A3D_ELEM(cudaMV, k, i, j - 1);
		if (i < FINISHINGX(cudaMV) && i > STARTINGX(cudaMV)) grad_y = CST(0.5) * A3D_ELEM(cudaMV, k, i + 1, j) - A3D_ELEM(cudaMV, k, i - 1, j);
		if (k < FINISHINGX(cudaMV) && k > STARTINGX(cudaMV)) grad_z = CST(0.5) * A3D_ELEM(cudaMV, k + 1, i, j) - A3D_ELEM(cudaMV, k - 1, i, j);

		// Compute magnitude of gradient vector
		PrecisionType magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z + CST(1e-5));


		if (j < FINISHINGX(cudaMV) && j > STARTINGX(cudaMV)) A3D_ELEM(cudaDx, k, i, j) = grad_x / magnitude;
		if (i < FINISHINGX(cudaMV) && i > STARTINGX(cudaMV)) A3D_ELEM(cudaDy, k, i, j) = grad_y / magnitude;
		if (k < FINISHINGX(cudaMV) && k > STARTINGX(cudaMV)) A3D_ELEM(cudaDz, k, i, j) = grad_z / magnitude;

		if (A3D_ELEM(cudaMV, k, i, j) > 0.0) A3D_ELEM(cudaDl1, k, i, j) = lst * CST(1.0);
		if (A3D_ELEM(cudaMV, k, i, j) < 0.0) A3D_ELEM(cudaDl1, k, i, j) = ll1 * CST(1.0) * A3D_ELEM(cudaMV, k, i, j);  // 0.1
	}	
}

template<typename PrecisionType>
__global__ void computeDTV(MultidimArrayCuda<PrecisionType> cudaReg, MultidimArrayCuda<PrecisionType> cudaDx,
                           MultidimArrayCuda<PrecisionType> cudaDy, MultidimArrayCuda<PrecisionType> cudaDz, MultidimArrayCuda<PrecisionType> cudaDl1,
						   const MultidimArrayCuda<int> VRecMaskB, double lambda, double ltv, double ltk, double ll1, double lst) {
	int cubeX = threadIdx.x + blockIdx.x * blockDim.x;
	int cubeY = threadIdx.y + blockIdx.y * blockDim.y;
	int cubeZ = threadIdx.z + blockIdx.z * blockDim.z;
	int k = STARTINGZ(cudaReg) + cubeZ;
	int i = STARTINGY(cudaReg) + cubeY;
	int j = STARTINGX(cudaReg) + cubeX;

    PrecisionType grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;
	PrecisionType grad_x2 = 0.0, grad_y2 = 0.0, grad_z2 = 0.0;

    // Compute gradients
	if (A3D_ELEM(VRecMaskB, k, i, j) != 0) {
		if (j < FINISHINGX(cudaReg) && j > STARTINGX(cudaReg)) grad_x = CST(0.5) * A3D_ELEM(cudaDx, k, i, j + 1) - A3D_ELEM(cudaDx, k, i, j - 1);
		if (i < FINISHINGX(cudaReg) && i > STARTINGX(cudaReg)) grad_y = CST(0.5) * A3D_ELEM(cudaDy, k, i + 1, j) - A3D_ELEM(cudaDy, k, i - 1, j);
		if (k < FINISHINGX(cudaReg) && k > STARTINGX(cudaReg)) grad_z = CST(0.5) * A3D_ELEM(cudaDz, k + 1, i, j) - A3D_ELEM(cudaDz, k - 1, i, j);
		if (j < FINISHINGX(cudaReg) && j > STARTINGX(cudaReg)) grad_x2 = CST(0.5) * A3D_ELEM(cudaDz, k, i, j + 1) * A3D_ELEM(cudaDx, k, i, j + 1) - A3D_ELEM(cudaDx, k, i, j - 1) * A3D_ELEM(cudaDx, k, i, j - 1);
		if (i < FINISHINGX(cudaReg) && i > STARTINGX(cudaReg)) grad_y2 = CST(0.5) * A3D_ELEM(cudaDy, k, i + 1, j) * A3D_ELEM(cudaDy, k, i + 1, j) - A3D_ELEM(cudaDy, k, i - 1, j) * A3D_ELEM(cudaDy, k, i - 1, j);
		if (k < FINISHINGX(cudaReg) && k > STARTINGX(cudaReg)) grad_z2 = CST(0.5) * A3D_ELEM(cudaDz, k + 1, i, j) * A3D_ELEM(cudaDz, k + 1, i, j) - A3D_ELEM(cudaDz, k - 1, i, j) * A3D_ELEM(cudaDz, k - 1, i, j);
		PrecisionType divergence = grad_x + grad_y + grad_z;
		PrecisionType divergence2 = CST(2.0) * (grad_x2 + grad_y2 + grad_z2);
		A3D_ELEM(cudaReg, k, i, j) = -CST(lambda) * (ltv * divergence + ltk * divergence2 + A3D_ELEM(cudaDl1, k, i, j));
	}	
}

}  // namespace cuda_forward_art_zernike3D
#endif	//CUDA_FORWARD_ART_ZERNIKE3D_CU
