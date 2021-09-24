% Compute min of 2-D function encountered in math-430 exam 3 #5
format long;

% Function f(x,y), but returns gradient too
function [z, grad] = fn_e3(x);
	z = [x(1).^2/7 - x(1)*11/5 + x(2).^2/5 - x(2)*17/6 + x(1)*x(2)/3 + 13];
	grad = [x(1)*2/7 - 11/5 + x(2)/3, x(2)*2/5 - 17/6 + x(1)/3];
end

f = @(x) [x(1).^2/7 - x(1)*11/5 + x(2).^2/5 - x(2)*17/6 + x(1)*x(2)/3 + 13];
F = @(x) [x(1)*2/7 - 11/5 + x(2)/3, x(2)*2/5 - 17/6 + x(1)/3];
tol = 1e-17;
max_iter = 100;
xinit = [-20;20];

% Custom method in newtons.m (shouldn't do well, since function is never 0)
fprintf(stdout, "Newtons:\n");
res = newtons(f, F, xinit, tol, max_iter);
res
f(res)

% Included minimizer #1
fprintf(stdout, "FMinsearch:\n");
opts = optimset("TolX", tol, "MaxIter", max_iter);
res = fminsearch(f, xinit, opts);
res
f(res)

% Included minimizer #2
fprintf(stdout, "FMinunc:\n");
opts = optimset("TolX", tol, "GradObj", "on", "MaxIter", max_iter);
res = fminunc(@fn_e3, xinit, opts);
res
f(res)

% Solve for root of gradient
fprintf(stdout, "System of gradient equations:\n");
D = [
	2/7 1/3;
	1/3 2/5;
	];
A = [
	11/5;
	17/6;
	];
res = D \ A;
res
f(res)

% True solution to above system of eqs
T = [-20.3;24]
f(T)
