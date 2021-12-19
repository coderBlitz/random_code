require("vector")

function pagerankpow(G, n)
	-- PAGERANKPOW PageRank by power method.
	-- x = pagerankpow(G) is the PageRank of the graph G.
	-- [x,cnt] = pagerankpow(G)
	--     counts the number of iterations.

	-- Link structure
	n = #G
	local L = {}
	local c = Vector:new(n)
	for j = 1,n do
		L[j] = {}
		c[j] = 0
		for k = 1,n do
			if (G[k][j] ~= 0) then
				L[j][(c[j] + 1)] = k
				c[j] = c[j] + 1
			end
		end
		L[j] = Vector:new(L[j])
	end

	-- Power method
	p = 0.85
	delta = (1-p) / n
	x = Vector:new(n, 1/n)
	z = Vector:new(n)
	cnt = 0
	while ((x-z):abs():max() > 0.000001) do
		z = x
		x = Vector:new(n)

		for j = 1,n	do
			if (c[j] == 0) then
				x = x + z[j] / n
			else
				for k = 1,c[j] do
					x[L[j][k]] = x[L[j][k]] + z[j]/c[j]
				end
			end
		end
		x = p*x + delta
		cnt = cnt+1
	end

	return x, cnt
end


-- Actual code
n = 6
G = {
	{0, 0, 0, 1, 0, 1},
	{1, 0, 0, 0, 0, 0},
	{0, 1, 0, 0, 0, 0},
	{0, 1, 1, 0, 0, 0},
	{0, 0, 1, 0, 0, 0},
	{1, 0, 1, 0, 0, 0}
}

x,cnt = pagerankpow(G)
print("Cnt:", cnt)

for _,v in ipairs(x) do
	print(v)
end
