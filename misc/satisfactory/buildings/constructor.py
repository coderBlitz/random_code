import copy as cp

class Constructor:
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
		"iron_plate": {"input": {"iron_ingot": 30}, "output": {"iron_plate": 20}},
		"iron_rod": {"input": {"iron_ingot": 15}, "output": {"iron_rod": 15}},
		"wire": {"input": {"copper_ingot": 15}, "output": {"wire": 30}},
		"copper_sheet": {"input": {"copper_ingot": 20}, "output": {"copper_sheet": 10}},
		"cable": {"input": {"wire": 60}, "output": {"cable": 30}},
		"concrete": {"input": {"limestone": 45}, "output": {"concrete": 15}},
		"screw": {"input": {"iron_rod": 10}, "output": {"screw": 40}},
		"biomass_l": {"input": {"leaves": 120}, "output": {"biomass_l": 60}},
		"biomass_w": {"input": {"wood": 60}, "output": {"biomass_w": 300}},
		"biomass_m": {"input": {"mycelia": 150}, "output": {"biomass_m": 150}},
		"biomass_a": {"input": {"alien_carapace": 15}, "output": {"biomass_a": 1500}},
		"solid_biofuel_l": {"input": {"biomass_l": 120}, "output": {"solid_biofuel_l": 60}},
		"solid_biofuel_w": {"input": {"biomass_w": 120}, "output": {"solid_biofuel_w": 60}},
		"solid_biofuel_m": {"input": {"biomass_m": 120}, "output": {"solid_biofuel_m": 60}},
		"solid_biofuel_a": {"input": {"biomass_a": 120}, "output": {"solid_biofuel_a": 60}},
		"power_shard_1": {"input": {"green_power_slug": 6}, "output": {"power_shard_1": 6}},
		"power_shard_2": {"input": {"yellow_power_slug": 4}, "output": {"power_shard_2": 8}},
		"power_shard_3": {"input": {"purple_power_slug": 3}, "output": {"power_shard_3": 15}},
		"steel_beam": {"input": {"steel_ingot": 60}, "output": {"steel_beam": 15}},
		"steel_pipe": {"input": {"steel_ingot": 30}, "output": {"steel_pipe": 20}},
		"color_cartridge": {"input": {"flower_petals": 37.5}, "output": {"color_cartridge": 75}},
		"spiked_rebar": {"input": {"iron_rod": 15}, "output": {"spiked_rebar": 15}},
		"quickwire": {"input": {"caterium_ingot": 12}, "output": {"quickwire": 60}},
		"silica": {"input": {"raw_quartz": 22.5}, "output": {"silica": 37.5}},
		"quartz_crystal": {"input": {"raw_quartz": 37.5}, "output": {"quartz_crystal": 22.5}}
	}
