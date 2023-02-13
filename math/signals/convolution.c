#include<math.h>
#include<stdio.h>
#include<stdlib.h>

/*/ Modified simpson's (odd N only)
//*/
double simpson(const double *const fn, const int N, const double h){
	double sum2 = 0;
	double sum4 = 0;

	for(int i = 1;i < (N-2);i += 2){
		sum4 += fn[i];
		sum2 += fn[i + 1];
	}

	//printf("Sum2: %.4lf\nSum4: %.4lf\n", sum2, sum4);

	const double sum = (fn[0] + 4*(sum4 + fn[N-2]) + 2*sum2 + fn[N-1]) * h/3;
	return sum;
}

void dump_signal(const char *const filename, double *sig, long N){
	FILE *fp = fopen(filename, "w");
	for(int i = 0;i < N;i++){
		fprintf(fp, "%.17lf\n", sig[i]);
	}
	fclose(fp);
}

int main(int argc, char *argv[]){
	const unsigned int rate = 1000; // Sample rate (Hz)
	const unsigned int N = 2 * rate;
	const unsigned int M = rate / 2;
	const unsigned int filter_length = 255;

	/*/ Generate signal data
	//*/
	double *input_sig = malloc(N * sizeof(*input_sig));
	if(input_sig == NULL){
		perror("Malloc failed");
		return 1;
	}

	const double freq = 4; // Frequency (Hz)
	for(int i = 0;i < N;i++){
		input_sig[i] = cos(2.0 * M_PI * freq * i / rate);
	}

	dump_signal("/tmp/input.txt", input_sig, N);

	/*/ Initialize convolution points (1Hz cosine in this case)
	//*/
	double *convolve_sig = malloc(filter_length * sizeof(*convolve_sig));
	if(input_sig == NULL){
		perror("Malloc failed");
		return 1;
	}

	// Initialize values, and sum to compute area (trapezoidal). Assumes odd filter length.
	double sum = 0.0;
	for(int i = 0;i < filter_length;i++){
		convolve_sig[i] = cos(M_PI * (i / (double) (filter_length-1) - 0.5));
		sum += cos(M_PI * (i / (double) (filter_length-1) - 0.5));
	}
	//sum = 2 * sum  / (double) (filter_length - 1); // Delta x (proper)
	printf("Sum/area = %.8lf\n", sum);

	for(int i = 0;i < filter_length;i++){
		convolve_sig[i] /= sum;
	}
	// No need to compute and divide by L-1 norm, as it would undo the area division (if done correctly)

	dump_signal("/tmp/convolve.txt", convolve_sig, filter_length);

	/*/ Do actual filter logic
	//*/
	// Output buffer
	double *output_sig = malloc(N * sizeof(*output_sig));
	if(output_sig == NULL){
		perror("Malloc failed");
		return 1;
	}
	// Scratch buffer
	double *scratch_sig = malloc(N * sizeof(*scratch_sig));
	if(scratch_sig == NULL){
		perror("Malloc failed");
		return 1;
	}

	// Calc inner-product the convolution with the signal
	double tmp;
	for(int n = filter_length;n < N;n++){
		sum = 0.0;
		for(int i = 0;i < filter_length;i++){
			tmp = input_sig[n - filter_length + i] * convolve_sig[i];
			sum += tmp;
			//scratch_sig[i] = tmp;
		}

		//output_sig[n] = sum;
		output_sig[n] = input_sig[n] - sum;
		//output_sig[n] = simpson(scratch_sig, filter_length, 1.0 / rate);
	}

	//*/
	dump_signal("/tmp/output.txt", output_sig, N);
	//*/

	/*/ Cleanup
	//*/
	free(scratch_sig);
	free(output_sig);
	free(convolve_sig);
	free(input_sig);

	return 0;
}
