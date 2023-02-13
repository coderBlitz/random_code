%%% Computes numerical derivatives using N points (x_N, and N-1 preceding points)
format long;

% Parameters
h = 1 / 10;
N = 5;
k = 1;

fig = 1;

% Derivation function
function y = dnxdt(x, N, h, k)
	c = [0:-1:1-N]; % Coefficients for each term: f(x - ch). So c = [0,-1,-2] ==> f(x) + f(x-h) + f(x-2h)
	ii = [0:(N-1)]';
	tab = inv(c .^ ii ./ factorial(ii)); % The inner calculation is a table of sum of taylor series components.
	s = flip(tab(:,k+1)'); % Extract the column based on which derivative is desired

	y = s * x ./ (h .^ k); % Do approximation calculation
end

%%% Actual stuff

xx = [0:h:3*pi];
yy = cos(xx);
dd = -sin(xx);

% Do calc
df = movfun(@(x)(dnxdt(x, N,h,k)), yy, [(N-1), 0], "Endpoints", "same");

df_error = df - dd;
error_mean = mean(df_error);
error_med = median(df_error);
error_std = std(df_error);

% Plot things
figure(fig++);
clf();
plot(xx, dd, xx, df);
legend("Actual", "Approximate");
xlabel("X");
ylabel("Y");
grid on;

figure(fig++);
clf();
semilogy(xx, abs(df_error));
legend("True error");
xlabel("X");
ylabel("Y");
grid on;
