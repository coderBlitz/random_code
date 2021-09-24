f = @(x) [x(1).^2 - 14*x(1)*x(2) + x(2).^2];
F = @(x) [2*x(1) - 14*x(2)	2*x(2) - 14*x(1)];

xinit = [1;2];
tol = 0.00000001;
max_iter = 40;

res = newtons(f, F, xinit, tol, max_iter);

disp(res);
printf("Err: %.8f\n", norm(f(res)));
