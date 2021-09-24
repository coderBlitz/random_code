#include<ctime>
#include<cmath>
#include<iostream>

using std::cout;
using std::endl;

// O(1) function
long con(long x){
	return x*x - 2*x + 1.0; // Help avoid minor optimization
}

// O(n) function
long lin(long x){
	long total = 0;
	for(long i = 0;i < x;i++){
		total += 2*i; // Not just i, to hopefully avoid optimization
	}
	return total;
}

// O(log_2 n) function
long logn(long x){
	double s = x;
	long c = 0;
	while(s > 1.0){
		s /= 2;
		c += 1;
	}
	return c;
}

// Arithmetic mean
double mean(double *arr, long n){
	if(arr == nullptr || n == 0) return NAN;

	double total = 0.0;
	for(long i = 0;i < n;i++){
		total += arr[i];
	}
	return total / n;
}

// Standard deviation
double stdev(double *arr, long n){
	if(arr == nullptr || n == 0) return NAN;

	double mu = mean(arr, n);
	double total = 0.0;
	for(long i = 0;i < n;i++){
		total += (mu - arr[i]) * (mu - arr[i]);
	}
	return sqrt(total / n);
}

int main(){
	const long M = 5; // Number of trials
	const long N = 1000; // Iterations per trial
	const long a = 2; // Scaling value
	double T[M]; // Result time array

	long (*fn)(long) = logn; // function to test
	// a^2 for linear, ~a for logarithmic, and a for constant
	double expect = a;

	/* Trial loop
	*/
	long S = N; // Scratch variable to leave N unchanged
	long res;
	long a_it = a; // Used to avoid calling pow()
	clock_t start,stop;
	for(long k = 0;k < M;k++){
		cout << "Trial " << k+1 << " with N = " << S << endl;

		// Call loop
		start = clock();
		for(long i = 0;i < S;i++){
			res = fn(i);
		}
		stop = clock();

		// Get results
		T[k] = (stop - start);

		// Iterate things
		S = N * a_it;
		a_it *= a;
	}

	// T array dump
	cout << "T =" << endl;
	for(long i = 0;i < M;i++){
		cout << "  " << T[i] << endl;
	}

	// Calculate ratios
	double B[M-1];
	for(long k = 0;k < (M-1);k++){
		B[k] = (T[k+1] / T[k]);
	}

	// B array dump
	cout << "B =" << endl;
	for(long i = 0;i < M-1;i++){
		cout << "  " << B[i] << endl;
	}

	// Calculate the suggested values
	double mu = mean(B, M-1);
	double dev = stdev(B, M-1);
	cout << "Mean = " << mu << endl;
	cout << "Std dev = " << dev << endl;


	// Values subject to vary (wildly) depending on system and test
	double mu_eps = 0.5;
	double dev_eps = mu_eps;
	bool pass = abs(mu - expect) < mu_eps && dev < dev_eps;

	cout << "Test " << ((pass) ? "succeeded!" : "FAILED!") << endl;
}
