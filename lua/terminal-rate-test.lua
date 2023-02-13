local counter = 0
local last = os.clock()

while true do
	counter = counter + 1

	local current = os.clock()

	io.write(string.format("Rate = %8d\r", counter))

	if (current - last) >= 1 then
		counter = 0
		last = current
		print()
	end
end
