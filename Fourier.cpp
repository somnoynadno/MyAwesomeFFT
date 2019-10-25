#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#define ITER_NUM 30

#include <bits/stdc++.h> // for fft
#include <chrono> // to work with time
#include <stdexcept> // to throw exception
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

    omp_fft(in, out, n, 2 * k);
    omp_fft(in + k, out + n, n, 2 * k);

#pragma omp parallel
    {
    #pragma omp for
        for(int i = 0, j = 0; i < n; i++, j += t) {
            ftype t = w[j] * out[i + n];
            out[i + n] = out[i] - t;
            out[i] += t;
        }
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

void printMyPlatforms();

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
    int t = 0;

    cout << "Current number of elements is " << n << endl;
    cout << "Current number of iteration is " << ITER_NUM << endl;

    vector<ftype> inv(n);
    cout << "Start FFT" << endl;
    
    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft(input.data(), inv.data(), n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "FFT ended with " << t/ITER_NUM << " ms" << endl;

    t = 0;
    vector<ftype> inv2(n);
    cout << "Start FFT with OpenMP" << endl;

    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        omp_fft(input.data(), inv2.data(), n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "FFT with OpenMP ended with " << t/ITER_NUM << " ms" << endl;

    t = 0;
    __m128d inv3[n];
    cout << "Start FFT with vectorization" << endl;

    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft_with_vec(input.data(), inv3, n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "FFT with vectorization ended with " << t/ITER_NUM << " ms" << endl;

    printMyPlatforms();

    return 0;
}
 

void printMyPlatforms(){
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

    for (i = 0; i < platformCount; i++) {

        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("%d. Device: %s\n", j+1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
            printf(" %d.%d Hardware version: %s\n", j+1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
            printf(" %d.%d Software version: %s\n", j+1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*) malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", j+1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                    sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", j+1, 4, maxComputeUnits);

        }

        free(devices);

    }

    free(platforms);
}