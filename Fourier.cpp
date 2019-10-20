#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <bits/stdc++.h> // for fft
#include <x86intrin.h> // for vectorization
#include <omp.h> // OpenMP, obviously
#include <CL/cl2.hpp> // OpenCL, obviously

using namespace std;

/*
COMPILER OPTIONS:
g++ Fourier.cpp -o Fourier -pthread -l OpenCL && ./Fourier
*/

typedef complex<double> ftype;
const double pi = acos(-1);
const int maxn = 1024;
ftype w[maxn];

// generate roots of -1
void init() {
    for(int i = 0; i < maxn; i++) {
        w[i] = polar(1., 2 * pi / maxn * i);
    }
}
 
template<typename T>
void fft(T *in, ftype *out, int n, int k = 1) {
    if(n == 1) {
        *out = *in;
        return;
    }
    int t = maxn / n;
    n >>= 1;
    fft(in, out, n, 2 * k);
    fft(in + k, out + n, n, 2 * k);
    for(int i = 0, j = 0; i < n; i++, j += t) {
        ftype t = w[j] * out[i + n];
        out[i + n] = out[i] - t;
        out[i] += t;
    }
}

template<typename T>
void omp_fft(T *in, ftype *out, int n, int k = 1) {
    if(n == 1) {
        *out = *in;
        return;
    }
    int t = maxn / n;
    n >>= 1;
#pragma omp parallel
    {
        fft(in, out, n, 2 * k);
        fft(in + k, out + n, n, 2 * k);
    }
    for(int i = 0, j = 0; i < n; i++, j += t) {
        ftype t = w[j] * out[i + n];
        out[i + n] = out[i] - t;
        out[i] += t;
    }
}
 
signed main() {
    // get the faster IO
    ios::sync_with_stdio(0);
    cin.tie(0);
    init();

    int n = maxn;
    vector <float> input; 

    // to generate random signal
    // for (int i = 0; i < n; i++){
    //     // r = random(0, 1)
    //     float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    //     input.push_back(r);
    // }

    for (int i = 0; i < 4*64; i++){
        input.push_back(0);
    }

    for (int i = 0; i < 8*64; i++){
        input.push_back(1);
    }

    for (int i = 0; i < 4*64; i++){
        input.push_back(0);
    }

    vector<ftype> inv(n);
    omp_fft(input.data(), inv.data(), n);

    // for (int i = 0; i < n; i++){
    //     cout << inv[i] << endl;
    // }

    return 0;
}
 