from typing import Listimport pprintclass Solution:
	def numIslands(self, grid: List[List[str]]) -> int:		count = 0		m = len(grid)		n = len(grid[0])		# First entry		if grid[0][0] != "0":			count = 1			grid[0][0] = 1		# Check rest of first row		for j in range(1, n):			# Only check non-zero entries			if grid[0][j] != "0":				# If left is non-zero, assign same island. Else increment, and assign new island				if grid[0][j-1] != "0":					grid[0][j] = grid[0][j-1]				else:					count += 1					grid[0][j] = count		# Check rest of grid		for i in range(1,m):			if grid[i][0] != "0":				# If top is non-zero, assign same island. Else increment, and assign new island
				if grid[i-1][0] != "0":
					grid[i][0] = grid[i-1][0]
				else:
					count += 1
					grid[i][0] = count			for j in range(1,n):				# Only check non-zero entries
				if grid[i][j] != "0":
					# If both, assign top to left and current, fix count. If one, assign current. Else increment
					if grid[i-1][j] != grid[i][j-1] and grid[i][j-1] != "0" and grid[i-1][j] != "0":
						grid[i][j] = grid[i-1][j]						grid[i][j-1] = grid[i-1][j]						count -= 1 # Converting two separate islands into one, decrement total by 1.					elif grid[i][j-1] != "0":						grid[i][j] = grid[i][j-1]					elif grid[i-1][j] != "0":
						grid[i][j] = grid[i-1][j]
					else:
						count += 1
						grid[i][j] = count		pp = pprint.PrettyPrinter(indent=2)		pp.pprint(grid)		return counts = Solution()# 1 islandgrid1 = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]# 3 islandsgrid2 = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]# 3 islandsgrid3 = [  ["0","1","0","0","1","0","1"],
  ["1","1","0","0","0","1","1"],]# 11 islandsgrid4 = [
  ["0","1","0","0","1","0","0","1"],
  ["1","1","0","1","1","0","1","1"],  ["0","0","1","0","0","1","0","0"],  ["0","1","1","0","1","1","0","0"],  ["0","0","0","1","0","0","1","0"],
  ["0","0","1","1","0","1","1","0"],  ["0","1","0","0","1","0","0","1"],
  ["1","1","0","1","1","0","1","0"]
]print(s.numIslands(grid1))print(s.numIslands(grid2))print(s.numIslands(grid3))print(s.numIslands(grid4))