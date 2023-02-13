seq = "UU";

% How many constraints will there be
n = length(seq);
N = 2*n;

% Constraint matrix
cons = zeros(N,N+2);
for i = 1:n
	step = seq(i);

	% Constraint row, with fixed identity coefficients, and fixed coefficient for even/odd constraint
	cons(2*i - 1, i) = 1;
	cons(2*i - 1, n + i + 1) = -2;
	cons(2*i, i + 1) = 1;

	% Constants for end (

	% Set coefficient based on step taken
	if step == 'D'
		cons(2*i, i) = -0.5;
	else
		cons(2*i, i) = -3;

		% Constants for odd
		cons(2*i - 1, end) = 1;
		cons(2*i, end) = 1;
	end
end

disp("Finding first")

% Reduce the constraints as far as possible (rank should be N)
[cons_red,k] = rref(cons);
cons_red(1,end-1:end)

% Compute first few initial values
n = 5;
first = 0;

% IMPORTANT: It seems skip is exponentially proportional to how many downward steps there are.
% Specifically, k downward steps translates to a skip of 2^k
% If last step is up, appears that k increments by 1
skip = 0;

count = 0;
c = 1;
while skip == 0
	k = (cons_red(1,end) - c)/cons_red(1,end-1);

	if abs(k - round(k)) < 1e-9
		if first == 0
			first = c;
		else
			skip = c - first;
		end
	end

	c += 1;
end

skip
few = first + skip * [0:4]
