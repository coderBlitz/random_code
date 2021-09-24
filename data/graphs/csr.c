/***	Author: Chris Skane
		Created: 26 MAR 2020
		Description: Graph structure using the Compressed-Sparse-Row format
		Notes:
		- An edge with no vertices might be an edge with a self-vertex of weight 0 (or Inf or NaN).
		- An edge with no vertices might also not be allowed (CSR limitation)

		TODO: Add vertices/getVertices function
		TODO: Add updateEdge/updateWeight function
***/

#include"csr.h"

/*	Add to num_vert when allocating, simply a way to offset initial allocation
	 size without largely affecting
*/
#define NUM_EDGE_OFFSET 255

#define MAX(x,y) ((x) > (y) ? (x) : (y))

/**	Zeroes all struct entries
**/
void initGraph(struct csr_graph *graph){
	if(graph == NULL) return;
	graph->num_edges = 0;
	graph->num_verts = 0;
	graph->num_rows = 0;
	graph->vals = NULL;
	graph->cols = NULL;
	graph->rows = NULL;
}

void destroyGraph(struct csr_graph *graph){
	if(graph == NULL) return;
	graph->num_edges = 0;
	graph->num_verts = 0;
	graph->num_rows = 0;
	free(graph->vals);
	free(graph->cols);
	free(graph->rows);
}

long shiftUp(unsigned int *arr, size_t size, size_t offset){
	if(offset < 0 || size < 1) return 0;

	size_t i = size+1;
	size_t count = 1;
	while(--i > offset && count <= size){
		arr[i] = arr[size-count];
		++count;
	}
}

/**	Insert a directed edge between vertices v1 and v2. Bidirectional not implemented yet.
**/
long insertArc(struct csr_graph *graph, unsigned int v1, unsigned int v2, double weight){
	if(graph == NULL) return -1;

	// Check if the array needs to be allocated. Requires that EDGE_OFFSET is of form 2^n - 1
	int num_leads = __builtin_clzl(graph->num_edges + NUM_EDGE_OFFSET); // Returns 31/63 for x == 0
	int new_num_leads = __builtin_clzl(graph->num_edges + NUM_EDGE_OFFSET + 1);

	if(num_leads > new_num_leads){
		size_t alloc_size = (1 << (64 - new_num_leads)) * sizeof(*graph->vals);
		//printf("alloc_size: %lu\tnum_edge: %lu\n", alloc_size, graph->num_edges);
		double *vals_temp = realloc(graph->vals, alloc_size);
		if(vals_temp == NULL){
			fprintf(stderr, "Could not allocate more space for vertices\n");
			return -1;
		}
		graph->vals = vals_temp;

		alloc_size = (1 << (64 - new_num_leads)) * sizeof(*graph->cols);
		//printf("alloc_size2: %lu\n", alloc_size);
		unsigned int *cols_temp = realloc(graph->cols, alloc_size);
		if(cols_temp == NULL){
			fprintf(stderr, "Could not allocate more space for columns\n");
			return -1;
		}
		graph->cols = cols_temp;
	}

	unsigned int largest_vert = MAX(v1, v2);
	if(largest_vert >= graph->num_rows){
		// Allocate one extra space, which is just equal to current number of vertices
		unsigned int *rows_temp = realloc(graph->rows, (largest_vert+2) * sizeof(*graph->rows));
		if(rows_temp == NULL){
			fprintf(stderr, "Could not allocate more space for rows\n");
			return -1;
		}
		if(graph->rows == NULL) for(size_t i = 0;i < largest_vert+2;i++) rows_temp[i] = 0;
		graph->rows = rows_temp;
		for(size_t i = graph->num_rows;i <= largest_vert;i++){
			//printf("Extending");
			graph->rows[i] = graph->rows[graph->num_rows];
		}
		graph->num_rows = largest_vert+1;
		graph->rows[largest_vert+1] = graph->num_edges;
	}

	// Find index to insert
	size_t colIdx = graph->rows[v1];
	size_t nextIdx = graph->rows[v1+1];
	size_t i = colIdx;
	while(i < nextIdx){
		if(graph->cols[i] > v2) break;
		else if(graph->cols[i++] == v2) return -1; // Don't allow duplicates
	}

	// If the vertex does not exist, increase the vertex count
	size_t row_size = nextIdx - colIdx;
	if(!row_size) graph->num_verts++;
	//printf("Inserting at index %u\n", i);

	//shiftUpf(graph->vals, graph->num_edges, i);
	memmove(graph->vals + i + 1, graph->vals + i, (graph->num_edges - i) * sizeof(*graph->vals));
	//shiftUp(graph->cols, graph->num_edges, i);
	memmove(graph->cols + i + 1, graph->cols + i, (graph->num_edges - i) * sizeof(*graph->cols));

	graph->vals[i] = weight;
	graph->cols[i] = v2;
	graph->num_edges++;

	//printf("NextIdx: %u\tv1: %u\n", nextIdx, v1);
	for(i = v1+1;i <= graph->num_rows;++i){
		++graph->rows[i];
	}
	if(graph->rows[graph->num_rows] != graph->num_edges){
		printf("Weird case!\n");
	}
	//graph->rows[graph->num_rows] = graph->num_edges;

	return 0;
}

/**	Remove the given arc from the graph. Does not shrink arrays.
	Return -1 for error, 1 if arc doesn't exist, 0 if successful.
**/
int removeArc(struct csr_graph *graph, unsigned int v1, unsigned int v2){
	if(graph == NULL) return -1;

	// Check if either vertex is out of the graph
	unsigned int largest_vert = MAX(v1, v2);
	if(largest_vert >= graph->num_rows){
		return 1;
	}

	size_t row_size = graph->rows[v1+1] - graph->rows[v1];
	if(row_size == 0){
		return 1;
	}

	// Find target vertex
	// TODO: Use binary search
	for(unsigned int i = 0;i < row_size;++i){
		if(graph->cols[i] == v2){
			// Shift cols and vals down by 1, decrement num_edges
			memmove(graph->vals + i, graph->vals + i + 1, (graph->num_edges - i) * sizeof(*graph->vals));
			memmove(graph->cols + i, graph->cols + i + 1, (graph->num_edges - i) * sizeof(*graph->cols));
			--graph->num_edges;

			unsigned int j = v1;
			unsigned int *val = graph->rows + j;
			while(++j <= graph->num_rows){
				//printf("Val (%p): %u\n", val, *val);
				--(*(++val));
			}

			break;
		}
	}

	// If all arcs for row/vertex removed, eliminate vertex
	if(--row_size == 0) graph->num_verts--;

	return 0;
}

/**	Returns a pointer to the location in the graph array where the neighbors of
	 v1 start. The ret_size parameter will be set to the count of v1 neighbors.
	If an invalid parameter is given, ret_size will remain untouched, and the
	 function will return NULL.
**/
unsigned int * getNeighbors(struct csr_graph const *graph, unsigned int v1, size_t *ret_size){
	if(graph == NULL) return NULL;

	if(v1 >= graph->num_rows){
		return NULL;
	}

	if(ret_size != NULL) *ret_size = graph->rows[v1+1] - graph->rows[v1];
	return &graph->cols[graph->rows[v1]];
}

/**	Returns a pointer to the location in the graph array where the weights of
	 v1 neighbors start. The ret_size parameter will be set to the count of weights.
	If an invalid parameter is given, ret_size will remain untouched, and the
	 function will return NULL.
**/
double * getWeights(struct csr_graph const *graph, unsigned int v1, size_t *ret_size){
	if(graph == NULL) return NULL;

	if(v1 >= graph->num_rows){
		return NULL;
	}

	if(ret_size != NULL) *ret_size = graph->rows[v1+1] - graph->rows[v1];
	return &graph->vals[graph->rows[v1]];
}

/**	Print out all 3 arrays in the graph
**/
void dumpGraph(struct csr_graph *graph){
	if(graph == NULL) return;

	for(size_t i = 0;i < graph->num_edges;i++){
		printf("%.1f, ", graph->vals[i]);
	}
	printf("\n");
	for(size_t i = 0;i < graph->num_edges;i++){
		printf("%u, ", graph->cols[i]);
	}
	printf("\n");
	for(size_t i = 0;i <= graph->num_rows;i++){
		printf("%u, ", graph->rows[i]);
	}
	printf("\n");
}
