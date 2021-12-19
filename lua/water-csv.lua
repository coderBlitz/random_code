source = "8635027_20180101to20181231.csv"

lines = {}
n_lines = 0

-- Method #1: Use read("*all") and gmatch()
--[[
fp = io.open(source, "r")
if fp == nil then
	os.exit(1)
end
cont = fp:read("*all")
fp:close()

print(string.len(cont))

--[[
-- Use pattern matching to extract each line
idx = 1
while true do
	local nxt,_ = string.find(cont, "\n", idx)
	if nxt == nil then
		break
	end

	-- Extract line
	local line = string.sub(cont, idx, nxt-1)
	table.insert(lines, line)

	-- Increment stuff
	idx = nxt + 1
	n_lines = n_lines + 1
end

-- Method specifically gets the 2 fixed fields of the water level file.
-- Likely fastest of all methods to extract the field data specifically, as
--  opposed to an extra processing step after any of the other line methods.
for time,level in string.gmatch(cont, "(%S+)\t(%S+)") do
	table.insert(lines, {time, level})
	n_lines = n_lines + 1
end

for line in string.gmatch(cont, "([^\n]*)\n") do
	table.insert(lines, line)
	n_lines = n_lines + 1
end
--]]

-- Method #2: Use io.lines()
---[[
for line in io.lines(source) do
	--table.insert(lines, line)
	n_lines = n_lines + 1
	lines[n_lines] = line
end
--]]

print("Parsed", n_lines, "lines")
print(lines[1])
print(lines[n_lines])
