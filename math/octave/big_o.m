format long g;

function ret = add(mat);
	ret = 0;
	%for i=1:size(mat)(1)
		%for j=1:size(mat)(2)
		%	ret += mat(i,j);
		%endfor
	%	ret += sum(mat(i,:));
	%endfor
	ret += sum(sum(mat));
end;

asymptotics = @(x)([x.^0 log(x) sqrt(x) x x.*log(x) x.^2 x.^2.*log(x) x.^3]);

n = [100*(1:8)]'
base = asymptotics(n);

for i=1:8
	s = n(i);
	mat = ones(s,s) * i;
	start = clock();
	res = add(mat)
	stop = clock();
	timed(i) = etime(stop,start);
endfor

disp("The times will be as follows:");
disp(timed');

best = base\timed'

pick = max(best)
