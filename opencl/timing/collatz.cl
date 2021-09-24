__kernel void collatz(__global unsigned int *results){
	unsigned long id = get_global_id(0);
	unsigned int count = 0; // Running total for compute loop

	unsigned long N = id;
	while(N > 1){
		// If results are zeroed, then add result to count if non-zero
		if((N & 0x1) == 0) N = N >> 1;
		else N = (N << 1) + N + 1;
		count++;
	}

	results[id] = count;// Store result in array
}
