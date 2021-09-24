from time import time
from random import randrange,seed

"""	Compressed sparse row graph class
"""
class CSRGraph:
	def __init__(self):
		self.num_verts = 0
		self.num_edges = 0
		self.num_rows = 0
		self.vals = []
		self.cols = []
		self.rows = []

	"""	Insert a directed edge between vertices v1 and v2. Bidirectional not implemented yet.
	"""
	def insertEdge(self, v1: int, v2: int, weight: float):
		# Extend if necessary
		largest_vert = max(v1, v2)
		if largest_vert >= self.num_rows:
			extend = [self.num_edges for i in range(largest_vert + 2 - self.num_rows)]
			# If empty, can't remove last item
			if len(self.rows) > 0:
				self.rows.pop()
			self.rows.extend(extend)

			self.num_rows = largest_vert+1
			self.rows[largest_vert+1] = self.num_edges

		# Find index to insert
		colIdx = self.rows[v1]
		nextIdx = self.rows[v1+1]
		i = colIdx
		row = self.cols[colIdx:nextIdx]
		if v2 in row:
			return False

		self.vals.insert(i, weight)
		self.cols.insert(i, v2)
		self.num_edges += 1

		for i in range(v1+1, self.num_rows + 1):
			self.rows[i] += 1

		return True

	"""	Returns a pointer to the location in the graph array where the neighbors of
		 v1 start. The ret_size parameter will be set to the count of v1 neighbors.
		If an invalid parameter is given, ret_size will remain untouched, and the
		 function will return NULL.
	"""
	def getNeighbors(self, v1: int) -> list:
		if v1 >= self.num_rows:
			return []

		return self.vals[self.rows[v1]:self.rows[v1+1]]

	"""	Print out all 3 arrays in the graph
	"""
	def __str__(self):
		ret = ",".join([str(a) for a in self.vals]) + "\n"
		ret += ",".join([str(a) for a in self.cols]) + "\n"
		ret += ",".join([str(a) for a in self.rows])
		return ret


def checkSizes(graph, insert_count: list, N, offset):
	print("Checking row size and columns..")

	max_count = 0
	max_vert_diff = N - graph.num_rows + offset
	#print("Max diff:", max_vert_diff,"\tNum rows:", graph.num_rows, "\tN:", N)
	# For each row
	for i in range(N - max_vert_diff):
		if insert_count[i] > max_count:
			max_count = insert_count[i]

		# Check size
		row_size = graph.rows[i+1 + offset] - graph.rows[i + offset]
		if row_size != insert_count[i]:
			print("SIZE MISMATCH FOR ROW %u! Got: %u\texpected: %u"%(i, row_size, insert_count[i]))
			print("INDICES %2u and %2u"%(i+1+offset, i+offset))
	print("Check done. Max count: %u"%max_count)

	# Check NNZ entry at end of row array
	idx = N + offset - max_vert_diff
	if graph.rows[idx] != N:
		print("SIZE MISMATCH! Got: %u\t(%u)\texpected: %u\n"%(graph.rows[idx], graph.rows[N + offset], N))

def checkData(graph, insert_count: list, edge_check, N, offset):
	max_vert_diff = N - graph.num_rows + offset
	print("Num_rows: %u\tdiff: %u\n" % (graph.num_rows, max_vert_diff))

	for i in range(N - max_vert_diff):
		# Check data validity
		row_start = graph.rows[i + offset]

		# For each column
		for j in range(insert_count[i]):
			idx = row_start + j
			col = graph.cols[idx] - offset

			if graph.vals[idx] != edge_check[i][col]:
				print("VAL MISMATCH FOR (%u, %u). Got: %u\texpected: %u"%(i, col, graph.vals[idx], edge_check[i][col]))

def main():
	seed(time())

	gr = CSRGraph()

	N = 5000

	# Will be dense matrix that stores same information as CSR
	edge_check = [[0.0]*N for i in range(N)]

	# Count the size of each row during insert
	simple_count = [0] * N

	# Insert random vertex pairings, in range [offset, N + offset)
	offset = 1
	dup_count = 0
	i = 0
	while i < N:
		a = randrange(N) + offset
		b = randrange(N) + offset

		#print("Inserting (%d, %d)"%(a, b))
		ret = gr.insertEdge(a, b, a*b/2.0)
		#print(gr)
		if not ret:
			dup_count += 1
			continue

		# Truth data
		edge_check[a-offset][b-offset] = a*b/2.0
		simple_count[a - offset] += 1

		i += 1
	print("%u duplicates encountered.\n" % dup_count)

	#print(gr)

	#	Make sure that each row is the correct size, and that the row has the
	#	 correct data in each column.
	checkSizes(gr, simple_count, N, offset)
	#print("\n".join([str(a) for a in edge_check]))
	checkData(gr, simple_count, edge_check, N, offset)

	return True

main()
