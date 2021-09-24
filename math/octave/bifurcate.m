%pkg load parallel; % Not necessary presently

global Nplot;
Nplot = 100;
r = [2.5:0.001:4.0]';

function res = trial(r)
	Npre = 250;
	global Nplot;
	x = zeros(length(r),Nplot);

	x(:,1) = 0.5;

	for n = [1:Npre]
		x(:,1) = r .* x(:,1) .* (1 - x(:,1));
	end

	for n = [1:Nplot-1]
		x(:,n+1) = r .* x(:,n) .* (1 - x(:,n));
	end

	res = x;
endfunction

x = trial(r);
%x = pararrayfun(nproc, @trial, r);
xvals = (r .* ones(1,Nplot));
plot(xvals, x, '.', 'markersize', 2);

title('Bifurcation diagram of the logistic map');
xlabel('r');  ylabel('x_n');
set(gca, 'xlim', [2.5 4.0]);
hold off; 
