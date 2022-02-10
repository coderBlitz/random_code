--[[
Vector

Provides basic vector-vector and vector-scalar operations

--]]

Vector = {}

-- Create new vector table object, and return
function Vector:new(nv, z)
	-- If nv is number, initialize vector of size nz with z (default 0)
	-- If nv array, fill vector with values from array

	z = z or 0

	local v = {
		length = 0,
		norm = Vector.norm,
		abs = Vector.abs,
		max = Vector.max,
		min = Vector.min,
		data = {}
	}
	if (type(nv) == "number") then
		-- Fill vector with zeros
		for i = 1,nv do
			v.data[i] = z
		end
		v.length = nv
	elseif (type(nv) == "table") then
		-- If table is another vector, then copy
		if (type(nv.data) == "table") then
			return Vector:new(nv.data)
		else
			-- Fill with array data
			for i = 1,#nv do
				v.data[i] = nv[i]
			end
			v.length = #nv
		end
	end

	setmetatable(v, self)

	return v
end

-- Access values with [index]
function Vector:__index(z)
	assert(type(z) == "number", "Index must be a number")
	assert(z > 0, "Index must be strictly positive")

	return self.data[z]
end

-- Set values with [index]
function Vector:__newindex(z, v)
	assert(type(z) == "number", "Index must be a number")
	assert(z > 0, "Index must be strictly positive")
	assert(z <= self.length, "Index out of range")

	self.data[z] = v
end

function Vector:__tostring()
	return table.concat(self.data, " ")
end

-- Add vectors or constants
function Vector:__add(b)
	-- If given a constant first, flip the parameters before doing stuff
	if (type(self) == "number" and type(b) == "table") then
		local tmp = self
		self = b
		b = tmp
	end

	local v_new = Vector:new(self.data)
	if (type(b) == "number") then
		-- Add constant to each value
		for i = 1,v_new.length do
			v_new.data[i] = v_new.data[i] + b
		end
	elseif (type(b) == "table") then
		-- Assume vector type, and add each value
		assert(v_new.length == b.length, "Vectors differ in length!")

		for i = 1,v_new.length do
			v_new.data[i] = v_new.data[i] + b.data[i]
		end
	end

	return v_new
end

-- Not making subtration addition of negative, to avoid extra overhead
function Vector:__sub(b)
	-- If given a constant first, flip the parameters before doing stuff
	if (type(self) == "number" and type(b) == "table") then
		local tmp = self
		self = -b
		b = -tmp
	end

	local v_new = Vector:new(self.data)
	if (type(b) == "number") then
		-- Sub constant to each value
		for i = 1,v_new.length do
			v_new.data[i] = v_new.data[i] - b
		end
	elseif (type(b) == "table") then
		-- Assume vector type, and subtract each value
		assert(v_new.length == b.length, "Vectors differ in length!")

		for i = 1,v_new.length do
			v_new.data[i] = v_new.data[i] - b.data[i]
		end
	end

	return v_new
end

-- Negation
function Vector:__unm()
	local v_new = Vector:new(self.data)

	for i = 1,v_new.length do
		v_new.data[i] = -v_new.data[i]
	end

	return v_new
end

-- Scalar multiplication, or dot/inner product
function Vector:__mul(b)
	-- If given a constant first, flip the parameters before doing stuff
	if (type(self) == "number" and type(b) == "table") then
		local tmp = self
		self = b
		b = tmp
	end

	local v_new = Vector:new(self.data)
	if (type(b) == "number") then
		-- Scale each value
		for i = 1,v_new.length do
			v_new.data[i] = v_new.data[i] * b
		end
	elseif (type(b) == "table") then
		-- Assume vector type, and scale each value
		assert(v_new.length == b.length, "Vectors differ in length!")

		for i = 1,v_new.length do
			v_new.data[i] = v_new.data[i] * b.data[i]
		end
	end

	return v_new
end

-- Scalar division ONLY
function Vector:__div(b)
	local v_new = Vector:new(self.data)
	if (type(b) == "number") then
		-- Divide each value by constant
		for i = 1,v_new.length do
			v_new.data[i] = v_new.data[i] / b
		end
	end

	return v_new
end

-- Return standard L-2 norm
function Vector:norm()
	local s = 0
	for i = 1,self.length do
		s = s + self.data[i] * self.data[i]
	end

	return math.sqrt(s)
end

-- Absolute value of vector
function Vector:abs()
	local v_new = Vector:new(self.data)
	for i = 1,v_new.length do
		v_new.data[i] = math.abs(self.data[i])
	end

	return v_new
end

-- Max value of vector (plus index)
function Vector:max()
	local m = self.data[1]
	local k = 1
	for i = 2,self.length do
		if (self.data[i] > m) then
			m = self.data[i]
			k = i
		end
	end

	return m, k
end

-- Min value of vector (plus index)
function Vector:min()
	local m = self.data[1]
	local k = 1
	for i = 2,self.length do
		if (self.data[i] < m) then
			m = self.data[i]
			k = i
		end
	end

	return m, k
end

local orig_type = type
type = function(obj)
	local obj_type = orig_type(obj)
	if obj_type == "table" and getmetatable(obj) == Vector then
		return "vector"
	end
	return obj_type
end
