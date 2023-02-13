--[[
TODO: Make two separate bubble/trickle functions for min and max, then assign them to new_heap accordingly
--]]

local heap = {}
heap.MIN = true
heap.MAX = false

function heap.new(capacity, is_min)
	local new_heap = {}

	-- Set variables
	if is_min ~= nil then
		new_heap._is_min = is_min
	else
		new_heap._is_min = heap.MIN
	end
	new_heap._capacity = capacity
	new_heap.size = 0
	new_heap._data = {}

	-- Set functions
	new_heap.insert = heap.insert
	new_heap.pop = heap.pop
	new_heap.dump = heap.dump
	new_heap.clear = heap.clear

	-- TODO: If desired to hide/lock metatable, don't use the exported/returned table
	heap.__metatable = heap
	setmetatable(new_heap, heap)

	return new_heap
end

function heap.insert(_h, x)
	if _h.size >= _h._capacity then
		return false
	end

	-- Append data to array
	_h.size = _h.size + 1
	_h._data[_h.size] = x

	-- Bubble up operation
	heap.bubble(_h)

	return true
end

--[[ Returns top-most heap element (min/max), and re-heapify
--]]
function heap.pop(_h)
	if _h.size <= 0 then
		return nil
	end

	-- Grab top entry
	dat = _h._data[1]

	-- Move bottom to top
	_h._data[1] = _h._data[_h.size]
	_h._data[_h.size] = nil
	_h.size = _h.size - 1

	-- Trickle down
	heap.trickle(_h)

	return dat
end

function heap.dump(_h)
	out = table.concat(_h._data, "\n")
	print(out)
end


-- Swap child with parent as necessary
function heap.bubble(_h)
	local d = _h.size

	local tmp

	if _h._is_min then
		-- Bubble with min rule
		while d ~= 1 and _h._data[d] < _h._data[math.floor(d/2)] do
			-- Swap values
			tmp = _h._data[d]
			_h._data[d] = _h._data[math.floor(d/2)]
			_h._data[math.floor(d/2)] = tmp

			d = math.floor(d/2)
		end
	else
		-- Bubble with max rule
		while d ~= 1 and _h._data[d] > _h._data[math.floor(d/2)] do
			-- Swap values
			tmp = _h._data[d]
			_h._data[d] = _h._data[math.floor(d/2)]
			_h._data[math.floor(d/2)] = tmp

			d = math.floor(d/2)
		end
	end
end

function heap.clear(_h)
	_h.size = 0
	_h._data = {}
end

-- Swap parent with (smaller/larger) child as necessary
function heap.trickle(_h)
	local d = 1
	local l,r

	if _h._is_min then
		-- Swap parent with smaller child (min)
		while d <= math.floor(_h.size / 2) do
			l,r = 2*d, 2*d + 1

			-- Find smaller child
			local child
			if r > _h.size then
				child = l
			elseif _h._data[l] < _h._data[r] then
				child = l
			else
				child = r
			end

			-- Swap with child if greater
			if _h._data[d] > _h._data[child] then
				-- Swap with child
				local tmp = _h._data[d]
				_h._data[d] = _h._data[child]
				_h._data[child] = tmp

				d = child
			else
				-- No more swaps necessary
				break
			end
		end
	else
		-- Swap parent with larger child (max)
		while d <= math.floor(_h.size / 2) do
			l,r = 2*d, 2*d + 1

			-- Find larger child
			local child
			if r > _h.size then
				child = l
			elseif _h._data[l] > _h._data[r] then
				child = l
			else
				child = r
			end

			-- Swap with child if lesser
			if _h._data[d] < _h._data[child] then
				-- Swap with child
				local tmp = _h._data[d]
				_h._data[d] = _h._data[child]
				_h._data[child] = tmp

				d = child
			else
				-- No more swaps necessary
				break
			end
		end
	end
end

-- Metamethods

return heap
