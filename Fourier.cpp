#define ITER_NUM 2

#include <bits/stdc++.h> // for fft
#include <chrono> // to work with time
#include <omp.h> // OpenMP, obviously

using namespace std;

/*
COMPILER OPTIONS:
g++ Fourier.cpp -o Fourier -pthread -l OpenCL -mssse3 -fopenmp && ./Fourier
*/

typedef complex<double> ftype;
typedef struct { double r; double i; } w_type;

const double pi = acos(-1);
const long int maxn = 1024*16*16*16*4;

double input[maxn]; 
w_type w_new[maxn];
w_type inv4[maxn];

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
    fft_c_parallel(in, out, w_new, n, 2 * k);
    fft_c_parallel(in + k, out + n, w_new, n, 2 * k);

    int j = 0;

	if (n == maxn || n == maxn/2 || n == maxn/4){
	#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			w_type res = complex_mult(w_new[j], out[i + n]);
			out[i + n].r = out[i].r - res.r;
			out[i].r += res.r;
			out[i + n].i = out[i].i - res.i;
			out[i].i += res.i;
		#pragma omp atomic
			j += t;
		}
	} else{
		for (int i = 0, j = 0; i < n; i++, j += t) {
			w_type res = complex_mult(w_new[j], out[i + n]);
			out[i + n].r = out[i].r - res.r;
			out[i].r += res.r;
			out[i + n].i = out[i].i - res.i;
			out[i].i += res.i;
		}
    }
}


int main() {
    // get the faster IO
    ios::sync_with_stdio(0);
    cin.tie(0);

    int n = maxn;
    
    // to generate roots of -1
    for(int i = 0; i < maxn; i++) {
        ftype w = polar(1., 2 * pi / maxn * i);
        w_new[i].r = w.real();
        w_new[i].i = w.imag();
    }

    // to generate random signal
    for (int i = 0; i < n; i++){
        // r = random(0, 1000)
        double r = rand() % 1000;
        input[i] = r;
    }

    
    chrono::high_resolution_clock::time_point t1;
    chrono::high_resolution_clock::time_point t2;
    int t = 0;

    cout << "Current number of elements is " << n << endl;
    cout << "Current number of iteration is " << ITER_NUM << endl;

    t = 0;
    cout << "Start pure C FFT" << endl;

    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft_c(input, inv4, w_new, n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "Pure C FFT ended with " << t << " ms" << endl;


    t = 0;
    cout << "Start parallel pure C FFT" << endl;
    for (int i = 0; i < ITER_NUM; i++){
        t1 = chrono::high_resolution_clock::now();
        fft_c_parallel(input, inv4, w_new, n);
        t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::milliseconds> (t2 - t1).count();
        t += duration;
    }
    cout << "Parallel pure C FFT ended with " << t << " ms" << endl;

    return 0;
}
 