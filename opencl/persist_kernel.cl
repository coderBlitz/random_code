__kernel void persist(__global const int *A, __global int *B){
	int id = get_global_id(0);

	B[id] += A[id];
}
