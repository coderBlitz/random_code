import copy as cp

class Foundry:
	"""	__init__(clock=100, target=None)
		clock -- The percentage clock rate of the foundry, where clock in [0,250]
		target -- The target recipe, for quick access later
	"""
	def __init__(self, clock=100, target=None):
		if clock < 0 or clock > 250:
			raise ValueError()

		self.clock = clock

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
			self.input = None
			self.output = None
			return

		# Copy and scale input
		self.input = cp.deepcopy(self.recipes[key]["input"])
		for val in self.input.keys():
			self.input[val] *= self.clock / 100

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

		# Scale input
		for val in self.input.keys():
			self.input[val] *= self.clock / 100

		# Scale output
		for val in self.output.keys():
			self.output[val] *= self.clock / 100

	# Base recipe values at 100%
	recipes = {
		"steel_ingot": {
			"input": {"iron_ore": 45, "coal": 45},
			"output": {"steel_ingot": 45}
		},
		"aluminum_ingot": {
			"input": {"aluminium_scrap": 240, "silica": 140},
			"output": {"steel_ingot": 4}
		}
	}
