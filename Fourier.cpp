#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <bits/stdc++.h> // for fft
#include <chrono> // to work with time
#include <stdint.h> // fix-sized integers
#include <x86intrin.h> // for vectorization
#include <omp.h> // OpenMP, obviously
#include <CL/cl2.hpp> // OpenCL, obviously

using namespace std;

/*
COMPILER OPTIONS:
g++ Fourier.cpp -o Fourier -pthread -l OpenCL -mssse3 && ./Fourier
*/

typedef complex<double> ftype;
const double pi = acos(-1);
const int maxn = 1024*16*16;
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
        omp_fft(in, out, n, 2 * k);
        omp_fft(in + k, out + n, n, 2 * k);
    }
    for(int i = 0, j = 0; i < n; i++, j += t) {
        ftype t = w[j] * out[i + n];
        out[i + n] = out[i] - t;
        out[i] += t;
    }
}

// complex numbers multiplication
static __inline__ __m128d ZMUL(__m128d a, __m128d b){
    __m128d ar, ai;

    ar = _mm_movedup_pd(a);
    ar = _mm_mul_pd(ar, b);
    ai = _mm_unpackhi_pd(a, a);
    b = _mm_shuffle_pd(b, b, 1);
    ai = _mm_mul_pd(ai, b);

    return _mm_addsub_pd(ar, ai);
}


void fft_with_vec(double *in, __m128d *out, int n, int k = 1) {
    if(n == 1) {
        *out = _mm_set_sd(*in);
        return;
    }
    int t = maxn / n;
    n >>= 1;
    fft_with_vec(in, out, n, 2 * k);
    fft_with_vec(in + k, out + n, n, 2 * k);

    __m128d w0, res;

    for (int i = 0, j = 0; i < n; i++, j += t) {
        w0 = _mm_set_pd(w[j].imag(), w[j].real());
        res = ZMUL(out[i+n], w0);

        out[i + n] = _mm_sub_pd(out[i], res);
        out[i] += res;
    }
}

int main() {
    // get the faster IO
    ios::sync_with_stdio(0);
    cin.tie(0);
    init();

    int n = maxn;
    vector <double> input; 

    // to generate random signal
    for (int i = 0; i < n; i++){
        // r = random(0, 1000)
        double r = rand() % 1000;
        input.push_back(r);
    }

    // n = 1024;
    // for (int i = 0; i < 4*64; i++){
    //     input.push_back(0);
    // }

    // for (int i = 0; i < 8*64; i++){
    //     input.push_back(1);
    // }

    // for (int i = 0; i < 4*64; i++){
    //     input.push_back(0);
    // }

    chrono::high_resolution_clock::time_point t1;
    chrono::high_resolution_clock::time_point t2;

    cout << "Current number of elements is " << n << endl;

    vector<ftype> inv(n);
    cout << "Start FFT" << endl;

    t1 = chrono::high_resolution_clock::now();
    fft(input.data(), inv.data(), n);
    t2 = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
    cout << "FFT ended with " << duration << " ms" << endl;

    vector<ftype> inv2(n);
    cout << "Start FFT with OpenMP" << endl;

    t1 = chrono::high_resolution_clock::now();
    omp_fft(input.data(), inv2.data(), n);
    t2 = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
    cout << "FFT with OpenMP ended with " << duration << " ms" << endl;

    __m128d inv3[n];
    cout << "Start FFT with vectorization" << endl;

    t1 = chrono::high_resolution_clock::now();
    fft_with_vec(input.data(), inv3, n);
    t2 = chrono::high_resolution_clock::now();

    duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
    cout << "FFT with vectorization ended with " << duration << " ms" << endl;

    // for (int i = 0; i < 10; i++){
    //     cout << inv[i] << endl;
    //     cout << inv3[i][0] << " " << inv3[i][1] << endl;
    // }

    return 0;
}
 