#ifndef CSR_H
#define CSR_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

struct csr_graph{
	size_t num_edges;	// Number of arcs in the graph
	size_t num_verts;	// Number of vertices in the graph
	size_t num_rows;	// Total rows in the graph (highest vertex index)

	// Data arrays
	double *vals;
	unsigned int *cols;
	unsigned int *rows;
};

/**	Zeroes all struct entries
**/
void initGraph(struct csr_graph *graph);

void destroyGraph(struct csr_graph *graph);

/**	Insert a directed edge between vertices v1 and v2. Bidirectional not implemented yet.
**/
long insertArc(struct csr_graph *graph, unsigned int v1, unsigned int v2, double weight);

/**	Remove the given arc from the graph. Does not shrink arrays.
	Return -1 for error, 1 if arc doesn't exist, 0 if successful.
**/
int removeArc(struct csr_graph *graph, unsigned int v1, unsigned int v2);

/**	Returns a pointer to the location in the graph array where the neighbors of
	 v1 start. The ret_size parameter will be set to the count of v1 neighbors.
	If an invalid parameter is given, ret_size will remain untouched, and the
	 function will return NULL.
**/
unsigned int * getNeighbors(struct csr_graph const *graph, unsigned int v1, size_t *ret_size);

/**	Returns a pointer to the location in the graph array where the weights of
	 v1 neighbors start. The ret_size parameter will be set to the count of weights.
	If an invalid parameter is given, ret_size will remain untouched, and the
	 function will return NULL.
**/
double * getWeights(struct csr_graph const *graph, unsigned int v1, size_t *ret_size);

/**	Print out all 3 arrays in the graph
**/
void dumpGraph(struct csr_graph *graph);

#endif
