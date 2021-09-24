__kernel void nextGen(__global const char *cur, __global char *next, __global unsigned int *WIDTH){
	unsigned long id = get_global_id(0);
	size_t size = get_global_size(0);

	char count = 0;
	unsigned int W = *WIDTH;
	long row = (id / W)*W;
	long next_row = ((id / W + 1)*W) % size;
	long prev_row = ((id / W - 1)*W + size) % size;
	// Previous row
	count += cur[prev_row + ((id + W) % W)];
	count += cur[prev_row + ((id+1 + W) % W)];
	count += cur[prev_row + ((id-1 + W) % W)];
	// Current row
	count += cur[row + ((id+1) % W)];
	count += cur[row + ((id-1 + W) % W)];
	// Next row
	count += cur[next_row + ((id + W) % W)];
	count += cur[next_row + ((id+1 + W) % W)];
	count += cur[next_row + ((id-1 + W) % W)];

	if(cur[id]){
		if(count == 2 || count == 3) next[id] = 1;
		else next[id] = 0; // Dies from over/underpopulation
	}else{
		if(count == 3){
			next[id] = 1;
		}else next[id] = 0;
	}
}
