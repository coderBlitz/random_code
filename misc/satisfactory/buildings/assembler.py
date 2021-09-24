import copy as cp

class Assembler:
	"""	__init__(clock=100, target=None)
		clock -- The percentage clock rate of the constructor, where clock in [0,250]
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
		"reinforced_iron_plate": {"input": {"iron_plate": 30, "screw": 60}, "output": {"reinforced_iron_plate": 5}},
		"rotor": {"input": {"iron_rod": 20, "screw": 100}, "output": {"rotor": 4}},
		"modular_frame": {"input": {"reinforced_iron_plate": 3, "iron_rod": 12}, "output": {"modular_frame": 2}},
		"encased_industrial_beam": {"input": {"steel_beam": 24, "concrete": 30}, "output": {"encased_industrial_beam": 6}},
		"stator": {"input": {"steel_pipe": 20, "wire": 30}, "output": {"stator": 6}},
		"motor": {"input": {"rotor": 10, "stator": 10}, "output": {"motor": 5}},
		"fabric": {"input": {"mycelia": 15, "biomass": 30}, "output": {"fabric": 15}},
		"circuit_board": {"input": {"wire": 60, "plastic": 30}, "output": {"circuit_board": 10}},
		"ai_limiter": {"input": {"circuit_board": 10, "quickwire": 90}, "output": {"ai_limiter": 5}}
	}
