#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200

#define ITER_NUM 50

#include <bits/stdc++.h> // for fft
#include <chrono> // to work with time
#include <x86intrin.h> // for vectorization
#include <omp.h> // OpenMP, obviously
#include <CL/cl2.hpp> // OpenCL, obviously

using namespace std;

/*
COMPILER OPTIONS:
g++ Fourier.cpp -o Fourier -pthread -l OpenCL -mssse3 -fopenmp && ./Fourier
*/

typedef complex<double> ftype;
typedef struct { double r; double i; } w_type;
typedef struct { float r; float i; } f_type;

const double pi = acos(-1);
const long int maxn = 1024;

__m128d m_out[maxn];
double my_input[maxn]; 
w_type w_new[maxn];
w_type inv[maxn];

float f_my_input[maxn];
f_type f_w_new[maxn];
f_type f_inv[maxn];

ftype w[maxn];

// generate roots of -1
void init() {
    for(int i = 0; i < maxn; i++) {
        w[i] = polar(1., 2 * pi / maxn * i);
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


void fft_with_vec(double *in, __m128d *out, w_type *w_new, int n, int k = 1) {
    if(n == 1) {
        *out = _mm_set_sd(*in);
        return;
    }
    int t = maxn / n;
    n >>= 1;
    fft_with_vec(in, out, w_new, n, 2 * k);
    fft_with_vec(in + k, out + n, w_new, n, 2 * k);

    __m128d w0, res;

    for (int i = 0, j = 0; i < n; i++, j += t) {
        w0 = _mm_set_pd(w_new[j].i, w_new[j].r);
        res = ZMUL(out[i+n], w0);

        out[i + n] = _mm_sub_pd(out[i], res);
        out[i] += res;
    }
}

w_type complex_mult(w_type a, w_type b){
    w_type c;
    c.r = a.r*b.r - a.i*b.i;
    c.i = a.i*b.r + a.r*b.i;
    return c;
}

void fft_c(double *in, w_type *out, w_type *w_new, int n, int k = 1){
    if (n == 1){
        out[0].r = in[0];
        out[0].i = 0;
        return;
    }

    int t = maxn / n;
    n >>= 1;
    fft_c(in, out, w_new, n, 2 * k);
    fft_c(in + k, out + n, w_new, n, 2 * k);

    for (int i = 0, j = 0; i < n; i++, j += t) {
        w_type res = complex_mult(w_new[j], out[i + n]);
        out[i + n].r = out[i].r - res.r;
        out[i].r += res.r;
        out[i + n].i = out[i].i - res.i;
        out[i].i += res.i;
    }
}

void fft_c_parallel(double *in, w_type *out, w_type *w_new, int n, int k = 1){
    if (n == 1){
        out[0].r = in[0];
        out[0].i = 0;
        return;
    }

    int t = maxn / n;
    n >>= 1;
    fft_c(in, out, w_new, n, 2 * k);
    fft_c(in + k, out + n, w_new, n, 2 * k);

    int j = 0;

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        // #pragma ordered
        // {
            w_type res = complex_mult(w_new[j], out[i + n]);
            out[i + n].r = out[i].r - res.r;
            out[i].r += res.r;
            out[i + n].i = out[i].i - res.i;
            out[i].i += res.i;
        #pragma omp atomic
            j += t;
        // }
    }
}

void fft(double *in, ftype *out, int n, int k = 1) {
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

void printMyPlatforms();

int main() {
    // get the faster IO
    ios::sync_with_stdio(0);
    cin.tie(0);

    init();

    int n = maxn;

    // to generate roots of -1
    for(int i = 0; i < maxn; i++) {
        ftype w = polar(1., 2 * pi / maxn * i);
        w_new[i].r = w.real();
        w_new[i].i = w.imag();
    }

    // // to generate random signal
    // for (int i = 0; i < n; i++){
    //     // r = random(0, 1000)
    //     double r = rand() % 1000;
    //     my_input[i] = r;
    // }

    n = 1024;
    for (int i = 0; i < 4*64; i++){
        my_input[i] = 0;
    }

    for (int i = 4*64; i < 8*64 + 4*64; i++){
        my_input[i] = 1;
    }

    for (int i = 8*64 + 4*64; i < 8*64 + 4*64 + 4*64; i++){
        my_input[i] = 0;
    }

    chrono::high_resolution_clock::time_point t1;
    chrono::high_resolution_clock::time_point t2;
    int t;

    cout << "Current number of elements is " << n << endl;
    cout << "Current number of iterations is " << ITER_NUM << endl;

    ftype inv2[maxn];
    t = 0;
    cout << "Start FFT (C++)" << endl;

    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft(my_input, inv2, n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "FFT ended with " << t << " ms" << endl;

    for (int i = 0; i < 20; i++) cout << inv2[i].real() << " " << inv2[i].imag() << endl;

    t = 0;
    cout << "Start pure C FFT" << endl;

    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft_c(my_input, inv, w_new, n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "Pure C FFT ended with " << t << " ms" << endl;

    for (int i = 0; i < 20; i++) cout << inv[i].r << " " << inv[i].i << endl;

    t = 0;
    cout << "Start parallel pure C FFT" << endl;

    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft_c_parallel(my_input, inv, w_new, n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "Parallel pure C FFT ended with " << t << " ms" << endl;

    for (int i = 0; i < 20; i++) cout << inv[i].r << " " << inv[i].i << endl;

    t = 0;
    cout << "Start FFT with vectorization" << endl;

    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft_with_vec(my_input, m_out, w_new, n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "FFT with vectorization ended with " << t << " ms" << endl;

    for (int i = 0; i < 20; i++){
        cout << m_out[i][0] << " " << m_out[i][1] << endl;
    }

    // printMyPlatforms();
 
    /* OpenCL goes here */

    // cast doubles to floats
    for (int i = 0; i < maxn; i++){
        f_my_input[i] = (float) my_input[i];
        f_w_new[i].r = (float) w_new[i].r;
        f_w_new[i].i = (float) w_new[i].i;
        f_inv[i].r = (float) inv[i].r;
        f_inv[i].i = (float) inv[i].i;
    }


    const char *KernelSource = "                                               \n" \
    "#define maxn 1024*16*16                                                   \n" \
    "typedef struct { float r; float i; } f_type;                              \n" \
    "f_type complex_mult(f_type a, f_type b){                                  \n" \
        "f_type c;                                                             \n" \
        "c.r = a.r*b.r - a.i*b.i;                                              \n" \
        "c.i = a.i*b.r + a.r*b.i;                                              \n" \
        "return c;                                                             \n" \
    "}                                                                         \n" \
    "void fft_c(__global float *in, __global f_type *out, __global f_type *w_new, int n, int k){          \n" \
        "if (n == 1){                                                          \n" \
        "    out[0].r = in[0];                                                 \n" \
        "    out[0].i = 0;                                                     \n" \
        "    return;                                                           \n" \
        "}                                                                     \n" \
        "int t = maxn / n;                                                     \n" \
        "n >>= 1;                                                              \n" \
        "fft_c(in, out, w_new, n, 2 * k);                                      \n" \
        "fft_c(in + k, out + n, w_new, n, 2 * k);                              \n" \
        "for (int i = 0, j = 0; i < n; i++, j += t) {                          \n" \
        "    f_type res = complex_mult(w_new[j], out[i + n]);                  \n" \
        "    out[i + n].r = out[i].r - res.r;                                  \n" \
        "    out[i].r += res.r;                                                \n" \
        "    out[i + n].i = out[i].i - res.i;                                  \n" \
        "    out[i].i += res.i;                                                \n" \
        "}                                                                     \n" \
    "}                                                                         \n" \
    "__kernel void func(                                                       \n" \
    "   __global float *input,                                                 \n" \
    "   __global f_type *input2,                                               \n" \
    "   __global f_type *output ){                                             \n" \
    "       int n = maxn;                                                      \n" \
    "       fft_c(input, output, input2, n, 1);                                \n" \
    "}                                                                         \n";


    int err;                            // error code returned from api calls

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
    cl_mem input2;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    // Bind to platform
    cl_platform_id cpPlatform;        // OpenCL platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);

    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    // Create a compute context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "func", &err);
    if (!kernel || err != CL_SUCCESS)
    {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * maxn, NULL, NULL);
    input2 = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(f_type) * maxn, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(f_type) * maxn, NULL, NULL);
    if (!input || !output || !input2)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }   

    for (int i = 0; i < 1024; i++) cout << f_my_input[i] << endl;

    // Write our data set into the input array in device memory 
    err  = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * maxn, f_my_input, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(commands, input2, CL_TRUE, 0, sizeof(f_type) * maxn, f_w_new, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    // Set the arguments to our compute kernel
    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &input2);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    global = maxn;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    // Wait for the command commands to get serviced before reading back results
    clFinish(commands);

    // Read back the results from the device to verify the output
    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(f_type) * maxn, f_inv, 0, NULL, NULL );  
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    for (int i = 0; i < 1024; i++) cout << f_inv[i].r << " " << f_inv[i].i << endl;

    cout << "That's all, folks" << endl;

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