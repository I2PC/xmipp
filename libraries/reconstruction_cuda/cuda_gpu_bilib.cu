__device__
float bspline03(float x) {
    float Argument = fabsf(x);
    if (Argument < 1.f)
        return Argument * Argument * (Argument - 2.f) * 0.5f + 2.f / 3.f;
    else if (Argument < 2.f) {
        Argument -= 2.f;
        return Argument * Argument * Argument * (-1.f / 6.f);
    } else
        return 0.f;
}

__device__
double bspline03(double x) {
    double Argument = fabs(x);
    if (Argument < 1.0)
        return Argument * Argument * (Argument - 2.0) * 0.5 + 2.0 / 3.0;
    else if (Argument < 2.0) {
        Argument -= 2.0;
        return Argument * Argument * Argument * (-1.0 / 6.0);
    } else
        return 0.0;
}
