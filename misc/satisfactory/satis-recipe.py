from buildings import *
#from smelter import Smelter
#from constructor import Constructor
#from assembler import Assembler
#from manufacturer import Manufacturer
#from foundry import Foundry
from math import ceil,floor,gcd
import sys

def lcm(a, b):
	return (a * b) // gcd(a,b)

def which(target):
	building = None
	if target in Smelter():
		building = Smelter
	elif target in Foundry():
		building = Foundry
	elif target in Constructor():
		building = Constructor
	elif target in Assembler():
		building = Assembler
	elif target in Manufacturer():
		building = Manufacturer

	return building

"""
s = Smelter()
for r in s:
	print(r)

	print(s[r])
	s.setTarget(r)

	print(s.input)
	print(s.output)
	print(s[r])
"""

# Just a test of iterating through recipes
#c = Constructor()
#for r in c:
#	print(r)
#	print(c[r])

"""	Returns an array/dictionary mix with the ratios at each stage
"""
def line(building, clock, target, prefix=""):
	# Determine equipment if none given
	if building == None:
		res = which(target)
		if res == None:
			print("Unknown recipe:", target)
			raise ValueError
		building = res

	b = building(clock, target)
	dest_amt = b.output[target]

	# destination amount
	dest = dest_amt

	ratios = []
	ret = []
	# Find LCM for all sources
	for source in b.input:
		base_amt = b.input[source]
		print(prefix, "Source: ", source, "\tamt: ", base_amt, sep='')

		res = [base_amt,[]]

		# Determine equipment
		building = which(source)
		if building != None:
			res = line(building, clock, source, prefix+"  ")

		src_amt = res[0]

		#print(prefix, "Src: ", src_amt, "\tBase: ", base_amt, sep='')
		ratio = src_amt / base_amt
		print(prefix, "Ratio: ", ratio,sep='')

		#ret.extend(res[1])
		ratios.append({ratio: res[1]})

	ret.extend(ratios)
	return (dest, ret)

#target = "screw"
#line(Constructor, 100, target)
#print()

if len(sys.argv) > 1:
	target = sys.argv[1]
	print("Target:", target)

	try:
		res = line(None, 100, target)
	except:
		print("Please give a valid recipe.")
		sys.exit(1)
	#print(res)
else:
	sys.exit(0)

ratios = res[1]
#print("Ratios:", ratios)

"""	Given the ratios from line(), find the number of machines required for the
	 final stage of the target product.
"""
def split(ratios):
	ret = 1
	for rats in ratios:
		#print(rats)

		ratio = list(rats.keys())[0]
		#print(ratio)
		res = split(rats[ratio])

		scale = 1
		num = ratio
		while ceil(num) != floor(num):
			scale += 1
			num = ratio * scale
		num = int(num)
		#print("Ratio:", num, "/", scale, "\tlcm:", lcm(scale, res))
		tmp = lcm(scale,res)
		ret = lcm(num * (tmp // scale), ret)
		#print("Ret:", ret)

	return ret

print()
res = split(ratios)
print("Asm/Con/Smlt needed: ", res)
