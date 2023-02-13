print("Enter a number:")
a = io.read("*n") -- read a number
print("You have entered", a)
print("a^2", a^2)

print("The following arguments were given to lua:")

local i = 0
while arg[i] do
	print(string.format("[%2d]", i), arg[i])
	i = i + 1
end

print(arg)
print(arg.n)
for i,v in pairs(arg) do
	print(string.format("[%2d]", i), v)
end
