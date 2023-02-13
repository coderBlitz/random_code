fig = 1;
% Parameters
rate = 1000; % Assume 1KHz sample rate
M = rate/2;
n = 1:M;

%% Create custom filter(s)
global b;
global c;
global d;
f_len = 128; % Filter length - 1 (we want odd filters, with 0 included, so make this even)
x = -1:(2/f_len):1; % Fix x range

% Filter 1
%b = 1 - abs(x); % Triangular filter
%b = 2 - 2 .^ abs(x); % Exponential triangle
b = cos(x * pi / 2); % Signal
trapz(b)
trapz(x,b)
b = b / trapz(x,b);
norm(b,1)
b = b / norm(b,1);

% Filter 2
c = 1 - x .^2; % Quadratic
%c = 1 - sqrt(abs(x)); % Sharp convex triangle
%c = sqrt(1 - abs(x)); % Sharp concave triangle
%c = sqrt(1 - x .^ 2); % Circular
%c = 2 - log((e^2 - e) * abs(x) + e); % Logarithmic triangle
%c = 2 - 2 .^ abs(x); % Exponential triangle
trapz(c)
trapz(x,c)
c = c / trapz(x,c);
norm(c,1)
c = c / norm(c,1);

d = 2 - 2 .^ abs(x); % Exponential triangle
d = d / trapz(x,d);
d = d / norm(d,1);


function y = convf1(x);
	global b;
	y = b * x;
end
function y = convf2(x);
	global c;
	y = c * x;
end
function y = convf3(x);
	global d;
	y = d * x;
end

% Calculate response (Trailing convolution)
baseline = zeros(1, M);
response = zeros(1, M);
response2 = zeros(1, M);
response3 = zeros(1, M);
nn = [-f_len:M];
for i = n;
	sig = cos(2*pi*nn * i/rate);
	baseline(i) = norm(sig((f_len+2):end)); % Only keep the entries after "warm up" period
	response(i) = norm(sig((f_len+2):end) - movfun(@convf1, sig, [f_len, 0], "Endpoints", 0)((f_len+2):end));
	response2(i) = norm(sig((f_len+2):end) - movfun(@convf2, sig, [f_len, 0], "Endpoints", 0)((f_len+2):end));
	response3(i) = norm(sig((f_len+2):end) - movfun(@convf3, sig, [f_len, 0], "Endpoints", 0)((f_len+2):end));
end
figure(fig++);
clf;
plot(n, baseline, n, response, n, response2, n, response3);
grid on;
title("Filter response");
legend("Perfect filter", "Trigonometric", "Quadratic", "Exponential");
xlabel("Frequency (Hz)");
ylabel("Filtered signal L2 error");

% Demonstration 1
figure(fig++);
clf;
sig = cos(2*pi*nn * 5/rate);
filtered1 = movfun(@convf1, sig, [f_len, 0], "Endpoints", 0);
filtered2 = movfun(@convf2, sig, [f_len, 0], "Endpoints", 0);
plot(nn, sig, nn, filtered1, nn, filtered2);
grid on;
title("5 Hz signal");
legend("Unfiltered", "Triangular", "Quadratic");
xlabel("Time intervals (1-rate/2)");
ylabel("Signal amplitude");

% Demo 2
figure(fig++);
clf;
sig = cos(2*pi*nn * 10/rate);
filtered1 = movfun(@convf1, sig, [f_len, 0], "Endpoints", 0);
filtered2 = movfun(@convf2, sig, [f_len, 0], "Endpoints", 0);
plot(nn, sig, nn, filtered1, nn, filtered2);
grid on;
title("10 Hz signal");
legend("Unfiltered", "Triangular", "Quadratic");
xlabel("Time intervals (1-rate/2)");
ylabel("Signal amplitude");

% Demo 3
figure(fig++);
clf;
sig = cos(2*pi*nn * 30/rate);
filtered1 = movfun(@convf1, sig, [f_len, 0], "Endpoints", 0);
filtered2 = movfun(@mean, sig, [f_len, 0], "Endpoints", 0);
plot(nn, sig, nn, filtered1, nn, filtered2);
grid on;
title("30 Hz signal with mean");
legend("Unfiltered", "Triangular", "Mean");
xlabel("Time intervals (1-rate/2)");
ylabel("Signal amplitude");

% Demo 4
figure(fig++);
clf;
sig = (cos(2*pi*nn * 5/rate) + cos(2*pi*nn * 10/rate) + cos(2*pi*nn * 30/rate)) / 3;
sig_intended = (cos(2*pi*nn * 5/rate)) / 3;
filtered1 = movfun(@convf1, sig, [f_len, 0], "Endpoints", 0);
filtered2 = movfun(@mean, sig, [f_len, 0], "Endpoints", 0);
plot(nn, sig, nn, filtered1, nn, filtered2, nn, sig_intended);
grid on;
title("Mixed signal 5hz + 10hz + 30 hz");
legend("Unfiltered", "Triangular", "Mean", "Intended");
xlabel("Time intervals (1-rate/2)");
ylabel("Signal amplitude");
