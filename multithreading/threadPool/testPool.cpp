#include<iostream>
#include<chrono>
#include<thread>
#include<random>
#include"threads.h"

using std::cout;
using std::endl;

void hello(int id, int tmp){
	std::cout << "Hello from " << id << ": " << tmp << endl;
	return;
}

static inline double rng_d(std::mt19937 &rnd, unsigned int mx, double breadth, double offset){
	return ((double)rnd() / (double)mx) * breadth + offset; // [0,1] * breadth - offset == [offset, offset + breadth]
}

int N;
unsigned long size;
double *mat1;
double *mat2;
double *mat3;

/**	For resunting index idx, multiply necessary row/col mat1 by mat2
**/
void mat_mul_piece(int _tid, long idx){
	long row = idx / N;
	long col = idx % N;

	double sum = 0;
	long idx_a, idx_b;
	for(long i = 0;i < N;i++){
		idx_a = row * N + i;
		idx_b = N * i + col;

		sum += mat1[idx_a] * mat2[idx_b];
	}

	mat3[idx] = sum;
}

int main(int argc, char *argv[]){
	ThreadPool<long> tp;

	cout << tp.size() << " threads available" << endl;
	//std::this_thread::sleep_for(std::chrono::seconds(2));

	std::random_device rd;
	std::mt19937 rng(rd());
	const unsigned int rng_max = rng.max();
	const double breadth = 100.0; // Range of decimal number
	const double offset = -breadth / 2.0; // Lowest value

	/*for(int i = 0;i < 2*tp.size();i++){
		tp.push(hello, i);
	}*/
	tp.start();
	N = 2000;
	size = N*N;
	mat1 = new double[size];
	mat2 = new double[size];
	mat3 = new double[size];

	cout << "Initializing matrices.." << endl;
	for(long i = 0;i < size;i++){
		mat1[i] = rng_d(rng, rng_max, breadth, offset);
		mat2[i] = rng_d(rng, rng_max, breadth, offset);
	}

	cout << "Queueing jobs.." << endl;

	for(long i = 0;i < size;i++){
		//mat_mul_piece(i, i);
		tp.push(mat_mul_piece, i);
	}

	do{
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}while(tp.length() > 0);

/*	cout << "Input 1:" << endl;
	for(long i = 0;i < size;i += N){
		for(long j = 0;j < N;j++){
			cout << mat1[i+j] << '\t';
		}
		cout << endl;
	}
	cout << "Input 2:" << endl;
	for(long i = 0;i < size;i += N){
		for(long j = 0;j < N;j++){
			cout << mat2[i+j] << '\t';
		}
		cout << endl;
	}

	cout << "Res:" << endl;
	for(long i = 0;i < size;i += N){
		for(long j = 0;j < N;j++){
			cout << mat3[i+j] << '\t';
		}
		cout << endl;
	}*/

	cout << "end = " << mat3[size-1] << endl;

	delete [] mat1;
	delete [] mat2;
	delete [] mat3;

	//std::this_thread::sleep_for(std::chrono::seconds(2));

	tp.stop();

	return 0;
}
