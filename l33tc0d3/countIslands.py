from typing import List
	def numIslands(self, grid: List[List[str]]) -> int:
				if grid[i-1][0] != "0":
					grid[i][0] = grid[i-1][0]
				else:
					count += 1
					grid[i][0] = count
				if grid[i][j] != "0":
					# If both, assign top to left and current, fix count. If one, assign current. Else increment
					if grid[i-1][j] != grid[i][j-1] and grid[i][j-1] != "0" and grid[i-1][j] != "0":
						grid[i][j] = grid[i-1][j]
						grid[i][j] = grid[i-1][j]
					else:
						count += 1
						grid[i][j] = count
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
  ["1","1","0","0","0","1","1"],
  ["0","1","0","0","1","0","0","1"],
  ["1","1","0","1","1","0","1","1"],
  ["0","0","1","1","0","1","1","0"],
  ["1","1","0","1","1","0","1","0"]
]