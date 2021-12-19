% Compute integral [0,2pi] of u(t)v(t) by using composite trapezoidal:
% For integral [0,2pi] g(t) dt
% h [g(0) + sum_{i=1}^{2n} g(t_i)]

% Trigonometric polys coefficients (2n+1 total points):
% h [ sum_{i=0}^{2n} u(t_i) v(t_i) ]

% Function to implement transform
function [aj, bj] = fourier_transform(F, n)
	if ! exist("n")
		n = floor(length(F) / 2); % Use largest N
	end
	aj = zeros(1,n+1);
	bj = zeros(1,n);
	F = F(1:2*n+1);

	% Constants
	h = 2 * pi / (2*n + 1);

	% a_0 is just the given values integrated
	aj(1) = h * sum(F) / 2; % No divide by pi, since that is done again later

	% All t values from 1 to 2n
	ti = h .* [0:2*n];

	% j values to compute. Complex to give both cosine and sine values
	j = complex(0, [1:n]');

	% Calculate both cosine and sine at same time
	exps = exp(j .* ti);
	vals = h .* sum(F .* exps, 2);

	% TODO: Consider returning complex values, instead of two vectors
	aj(2:end) = real(vals) / pi;
	bj = imag(vals) / pi;
end
