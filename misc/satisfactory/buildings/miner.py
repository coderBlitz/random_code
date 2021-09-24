import copy as cp

"""	Miner
	Currently assumes an impure node (for smallest LCM)
"""
class Miner:
	"""	__init__(clock=100, target=None, level=1)
		clock -- The percentage clock rate of the miner, where clock in [0,250]
		target -- The target recipe, for quick access later
		level -- The level of the miner, 1 for Mk1, 2 for Mk2, 3 for Mk3
	"""
	def __init__(self, clock=100, target=None):
		if clock < 0 or clock > 250 or level < 1 or level > 3:
			raise ValueError()

		self.clock = clock
		self.level = level

		# Set recipe info
		self.setTarget(target)

	def __iter__(self):
		self.idx = 0
		self.recipe_keys = list(self.recipes.keys())
		return self

	def __next__(self):
		x = self.idx
		if x >= len(self.recipe_keys):
			raise StopIteration

		self.idx += 1
		return self.recipe_keys[x]

	def __getitem__(self, key):
		return self.recipes[key]

	"""	setTarget(key)
		key -- Update target recipe, and scale input and output to clock
	"""
	def setTarget(self, key):
		if key == None:
			self.target = None
			self.output = None
			return

		# Copy and scale output
		self.output = cp.deepcopy(self.recipes[key]["output"])
		for val in self.output.keys():
			self.output[val] *= self.clock / 100

		# Set target here (only runs if key is valid, otherwise above would error)
		self.target = key

	"""	setClock(clock)
		clock -- Update clock and rescale input and output
	"""
	def setClock(self, clock):
		if clock < 0 or clock > 250:
			raise ValueError()
		self.clock = clock

		# Scale output
		for val in self.output.keys():
			self.output[val] *= self.level * self.clock / 100

	# Base recipe values at 100%
	recipes = {
		"iron_plate": {"input": {"iron_ingot": 30}, "output": {"iron_plate": 20}},
	}
