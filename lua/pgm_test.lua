magic = "P5"
width = 110
height = 100
maxval = 255

out_filename = "/tmp/out.pgm"

-- TODO: Flatten loop
rows = {}
for i = 1,height do
	row = {}
	for j = 1,width do
		row[j] = math.floor(((i-1) / height) * maxval)
	end

	rows[i] = row
end

-- TODO: Flatten loop, as above. Remove table concat in loop, leave output concat only
dat_rows = {}
for i = 1,height do
	dat_row = {}
	for j = 1,width do
		dat_row[j] = string.pack("B", rows[i][j])
	end

	dat_rows[i] = table.concat(dat_row)
	--print("dat[i]:", dat_rows[i])
end

out_file = io.open(out_filename, "w")
if type(out_file) == nil then
	print("Error opening file")
	os.exit(1)
end
out_file:write(magic .. "\n") -- Write magic
out_file:write(width .. " " .. height .. " " .. maxval .. "\n") -- Write metadata
--[[
for i = 1,height do
	out_file:write(dat_rows[i]) -- Write each row data
end
--]]
out_file:write(table.concat(dat_rows))
out_file:close() -- Close
