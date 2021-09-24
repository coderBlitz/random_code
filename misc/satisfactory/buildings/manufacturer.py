import copy as cp

class Manufacturer:
	"""	__init__(clock=100, target=None)
		clock -- The percentage clock rate of the manufacturer, where clock in [0,250]
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
		"heavy_modular_frame": {
			"input": {"modular_frame": 10, "steel_pipe": 30, "encased_industrial_beam": 10, "screw": 180},
			"output": {"heavy_modular_frame": 2}
		},
		"computer": {
			"input": {"circuit_board": 18.75, "cable": 22.5, "plastic": 33.75, "screw": 112.5},
			"output": {"computer": 1.875}
		},
		"supercomputer": {
			"input": {"computer": 2, "ai_limiter": 3.75, "high_speed_connector": 5.625, "plastic": 39.375},
			"output": {"supercomputer": 1.875}
		},
		"high_speed_connector": {
			"input": {"quickwire": 100, "cable": 25, "plastic": 15},
			"output": {"high_speed_connector": 2.5}
		},
		"filter": {
			"input": {"coal": 37.5, "rubber": 15, "fabric": 15},
			"output": {"filter": 7.5}
		},
		"crystal_oscillator": {
			"input": {"quartz_crystal": 18.75, "cable": 26.25, "reinforced_iron_plate": 7.5},
			"output": {"crystal_oscillator": 1.875}
		},
		"nobelisk": {
			"input": {"black_powder": 15, "steel_pipe": 15, "beacon": 3},
			"output": {"nobelisk": 3}
		},
		"beacon": {
			"input": {"iron_plate": 22.5, "iron_rod": 7.5, "wire": 112.5, "cable": 15},
			"output": {"beacon": 7.5}
		}
	}
