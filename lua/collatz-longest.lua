high = 100000000
if (#arg >= 2) then
	high = tonumber(arg[1], 10)
end

lens = {0}
for i = 1,high do
	lens[i] = 0
end
max = 1

for i = 2,high do
	count = 0
	num = i
	while num > 1 do
		if (num % 2) == 0 then
			num = num / 2
		else
			num = num * 3 + 1
		end
		count = count + 1

		if num < i then
			-- Stop counting, cause we've already done it
			count = count + lens[num]
			break
		end
	end

	lens[i] = count
	if lens[i] > lens[max] then
		max = i
	end
end

print(string.format("Largest chain: %d (%u)\n", lens[max], max))
