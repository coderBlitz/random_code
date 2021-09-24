#include<stdio.h>
#include<stdlib.h>
#include"csr.h"
#include"../stack.h"
#include"../heaps/heap.h"

// To be used in heap
struct point{
	unsigned int idx;	// This vertex
	double weight;		// Current path weight
	unsigned int src;	// Previous vertex
};

/**	 Priority function for data pointers
**/
static inline int ptCmp(void *a, void *b){
	return ((struct point *)a)->weight < ((struct point *)b)->weight; // Min function for weights
}

/**	Reverse an array in place, by repeatedly swapping outer values and moving inward
**/
void reversel(long *arr, size_t N){
	if(arr == NULL || N < 2) return;

	size_t i = 0, j = N-1;
	long tmp;
	while(i < j){
		tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;

		++i;
		--j;
	}
}

/**	Performs dijkstra's algorithm on `graph`, from `src` to `dst`.
	Returns array of indices taken in the path (in order taken), with -1
	 denoting the end of the path.
**/
long * dijkstra(struct csr_graph *graph, long src, long dst){
	/*
	Dijkstra's algorithm:
		1. Create an array of visited vertices
		2. Assign each vertex an initial distance, 0 for start/current else inf/NaN
		3. Consider all unvisited neighbors, and calculate their distance through
		 the current node. Keep the smaller of the existing and calculated distance.
		4. Mark current node as visited.
		5. If the destination is marked visited, or if the smallest distance among
		 the unvisited set if infinity, then stop. The algorithm is done.
		6. Otherwise, set the current vertex as the unvisited vertex with the smallest
		 tenative distance and go to step 3.

	"Pseudo-code"/Plan (bold emphasis on structures):
		Create array the size of total vertices, initialize to all 0's
		TODO: (Maybe) Create array of POINTs for each vertex, then insert these pointers into the HEAP (weights INFINITY)
		Initialize HEAP, insert start/current node
		Create loop until HEAP is empty, or cur == dst
			Make current node the next node from the HEAP
			Mark current node as visited
			Create loop over all neighbors, skipping visited
				Calculate distance for neighbor
				Update else insert POINT into the HEAP
			Push POINT onto STACK
		Empty stack to reconstruct path
		Reverse path for correct order

	Data format/representation:
		* "Nodes" will be pointers to the data array in the graph structure `graph`
	*/

	if(graph == NULL || src == dst){
		fprintf(stderr, "Invalid parameters.\n");
		return NULL;
	}

	/* Setup necessary structures
	*/
	size_t num_verts = graph->num_rows; // 1:1 array requires the num_rows not num_verts
	char *visited = calloc(num_verts, sizeof(*visited)); // Boolean array for if a node was ever visited
	if(visited == NULL){
		return NULL;
	}
	struct point *points = malloc(num_verts * sizeof(*points)); // Will represent a point/node in the path
	if(points == NULL){
		return NULL;
	}

	struct heap hp;
	heap_init(&hp, num_verts, ptCmp); // Heap will store the points/nodes encountered while searching

	// Initialize the first (source) vertex, then insert to heap
	points[src].idx = src;
	points[src].weight = 0;
	points[src].src = src;

	heap_insert(&hp, points + src);
	//heap_dump(&hp);
	// TODO: If time preferred over space, add all points to heap with weight=INFINITY


	/* Search loop, while the heap is not empty
	*/
	// Scratch variables
	unsigned int cur_vtx;
	struct point *cur_pt;
	unsigned int *nbrs;
	size_t num_nbrs = 0;
	double *nbr_weights;
	double sum;
	unsigned int idx;
	long res;
	while(hp.size){
		// Get next point from heap, and mark vertex visited
		cur_pt = (struct point *) heap_pop(&hp);
		cur_vtx = cur_pt->idx;
		visited[cur_vtx] = 1;
		//printf("Cur_vtx: %u\t%f\n", cur_vtx, cur_pt->weight);

		// Early exit condition (when at destination
		if(cur_vtx == dst){
			printf("Found destination!\n");
			break;
		}

		// Get the neighbors and weights of the current vertex
		nbrs = getNeighbors(graph, cur_vtx, &num_nbrs);
		if(nbrs == NULL){
			fprintf(stderr, "Nbr error\n");
			continue;
		}
		nbr_weights = getWeights(graph, cur_vtx, NULL);
		if(nbr_weights == NULL){
			fprintf(stderr, "Weights error\n");
			continue;
		}

		/* Iterate over neighbors and update their weights.
			Inserts point into heap if it is not already there
		*/
		for(unsigned i = 0;i < num_nbrs;++i){
			if(visited[nbrs[i]]) continue; // Skip visited
			idx = nbrs[i];

			sum = cur_pt->weight + nbr_weights[i]; // Path cost to neighbor, through current node
			//printf("%2u -> %2u: %f\n", cur_vtx, idx, sum);

			// Set point info if not in heap (previously unset), else update weight if new path is less
			if(heap_update(&hp, points + idx) < 0){
				points[idx].idx = idx;
				points[idx].weight = sum;
				points[idx].src = cur_vtx;

				heap_insert(&hp, points + idx); // Insert new point
			}else if(!(points[idx].weight <= sum)){
				// Update weight if sum is less
				points[idx].weight = sum;
				points[idx].src = cur_vtx;

				heap_update(&hp, points + idx); // Update point
			}

			// Update else insert point into the heap (TODO: Remove when confident above change works)
			/*if(heap_update(&hp, points + idx) < 0){
				//printf("Inserting %2u into heap\n", idx);
				heap_insert(&hp, points + idx);
			}*/
		}

		// Put point into the finished stack
		//printf("Pushing %lu\n", cur_vtx);
		push(points + cur_vtx);
	}

	// cur_vtx should be == dst, if path exists
	if(cur_vtx != dst){
		printf("No path exists!\n");
		return NULL;
	}


	/* Reconstruct (reversed) path by emptying stack, then reverse
	*/	
	long *path = malloc(stack_count() * sizeof(*path));
	if(path == NULL){
		return NULL;
	}

	//printf("Emptying stack\n");
	size_t size = 1;
	path[0] = dst;
	while(stack_count()){
		cur_pt = (struct point *) pop();
		idx = cur_pt->idx;

		if(points[cur_vtx].src == idx){
			//printf("Adding %lu to path\n", idx);
			path[size++] = idx;
			cur_vtx = idx;
		}
	}
	path[size] = -1; // End of path marker

	// Print un-reversed path
	/*printf("Path length: %lu\n", size);
	for(int i = 0;i < size;++i){
		printf("%2u <- ", path[i]);
	}
	printf("\n");*/

	reversel(path, size);

	/* Clean up
	*/
	free(visited);
	free(points);
	heap_destroy(&hp);

	return path;
}

int main(){
	struct csr_graph gr;
	initGraph(&gr);

	// Initialize graph
	/*	Test small graph
	1 → 2
	↑   ↓ ↘
	3 → 4 → 5
	↓   ↑ ↗
	6 → 7
	*/
	/*insertArc(&gr, 1, 2, 1.0);
	insertArc(&gr, 2, 4, 1.0);
	insertArc(&gr, 2, 5, 2.0);
	insertArc(&gr, 3, 1, 1.0);
	insertArc(&gr, 3, 4, 3.0);
	insertArc(&gr, 3, 6, 1.0);
	insertArc(&gr, 4, 5, 1.0);
	insertArc(&gr, 6, 7, 1.0);
	insertArc(&gr, 7, 4, 1.0);
	insertArc(&gr, 7, 5, 1.0);*/

	/*	Computerphile graph (Corect answer is 19->2->8->7->5, or 'S'->'B'->'H'->'G'->'E')
	*/
	insertArc(&gr, 1, 2, 3.0); // AB
	insertArc(&gr, 1, 4, 4.0); // AD
	insertArc(&gr, 1, 19, 7.0); // AS
	insertArc(&gr, 2, 1, 3.0); // BA
	insertArc(&gr, 2, 19, 2.0); // BS
	insertArc(&gr, 2, 4, 4.0); // BD
	insertArc(&gr, 2, 8, 1.0); // BH
	insertArc(&gr, 3, 19, 3.0); // CS
	insertArc(&gr, 3, 12, 2.0); // CL
	insertArc(&gr, 4, 1, 4.0); // DA
	insertArc(&gr, 4, 2, 4.0); // DB
	insertArc(&gr, 4, 6, 5.0); // DF
	insertArc(&gr, 5, 7, 2.0); // EG
	insertArc(&gr, 5, 11, 5.0); // EK
	insertArc(&gr, 6, 4, 5.0); // FD
	insertArc(&gr, 6, 8, 3.0); // FH
	insertArc(&gr, 7, 8, 2.0); // GH
	insertArc(&gr, 7, 5, 2.0); // GE
	insertArc(&gr, 8, 6, 3.0); // HF
	insertArc(&gr, 8, 2, 1.0); // HB
	insertArc(&gr, 8, 7, 2.0); // HG
	insertArc(&gr, 9, 12, 4.0); // IL
	insertArc(&gr, 9, 10, 6.0); // IJ
	insertArc(&gr, 9, 11, 4.0); // IK
	insertArc(&gr, 10, 11, 4.0); // JK
	insertArc(&gr, 10, 12, 4.0); // JL
	insertArc(&gr, 10, 9, 6.0); // JI
	insertArc(&gr, 11, 9, 4.0); // KI
	insertArc(&gr, 11, 10, 4.0); // KJ
	insertArc(&gr, 11, 5, 5.0); // KE
	insertArc(&gr, 12, 3, 2.0); // LC
	insertArc(&gr, 12, 9, 4.0); // LI
	insertArc(&gr, 12, 10, 4.0); // LJ
	insertArc(&gr, 19, 1, 7.0); // SA
	insertArc(&gr, 19, 2, 2.0); // SB
	insertArc(&gr, 19, 3, 3.0); // SC

	printf("Graph:\n");
	dumpGraph(&gr);

	printf("Solving..\n");
	long *res = dijkstra(&gr, 19, 5);

	size_t size = 0;
	if(res != NULL){
		while(res[size] != -1) ++size;
	}else{
		printf("Dijkstra error.\n");
	}

	printf("Path:\n");
	for(int i = 0;i < size;++i){
		printf("%2u ->", res[i]);
	}
	printf(" QED\n");

	destroyGraph(&gr);
	free(res);
}
