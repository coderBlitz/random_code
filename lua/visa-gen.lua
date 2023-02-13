-- Multiplying by 2, then summing resulting digits yields the following pattern.
-- Avoids lots of string processing and extra math
lookup = {
	["0"] = 0,
	["1"] = 2,
	["2"] = 4,
	["3"] = 6,
	["4"] = 8,
	["5"] = 1, -- 10 --> 1 + 0 = 1
	["6"] = 3, -- 12 --> 1 + 2 = 3
	["7"] = 5, -- ...
	["8"] = 7,
	["9"] = 9
}

-- Perform Luhn's algorithm (TODO: Modify to work with variable digits)
function verify(num)
	local sum = 0
	local fn = num:gmatch("(%d)")
	local l = fn()

	while l ~= nil do
		sum = sum + lookup[l]

		l = fn()
		sum = sum + l

		l = fn()
	end

	return (sum % 10) == 0, sum
end

-- Generate random (valid) credit card numbers
--[[
IIN of major credit orgs (leading digits):
	Visa: 4
	AMEX: 34, 37 (ONLY 15 DIGITS TOTAL)
	Discover: 6011, 622126-622925, 624000-626999, 628200-628899, 64, 65
	Mastercard: 2221-2720, 51-55
--]]

function genNum()
	-- Generate the random string of 15 digits
	local num = ""
	for i = 1,15 do
		num = num .. math.random(0,9)
	end

	-- Append 0 to get sum
	local temp  = num .. "0"

	res,sum = verify(temp)

	-- If necessary, compute proper checksum digit, then append
	if not res then
		last = 10 - (sum % 10)
		num = num .. last
	else
		num = temp
	end

	return num
end

math.randomseed(os.time())

test = {
	"4578 4230 1376 9219",
	"9999 9999 9999 9995"
}

for _,num in ipairs(test) do
	print(verify(num))
end

n = 10
for i = 1,n do
	print(genNum())
end
