__kernel void vector_add(__global float *A){
	int id = get_global_id(0);
	if(A[id]) A[id] = -A[id];
}
