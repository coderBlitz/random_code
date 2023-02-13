files = {"/tmp/input.txt", "/tmp/output.txt", "/tmp/convolve.txt"};

fig = 1;
for i = 1:length(files)
	figure(fig++);
	clf;
	dat = dlmread(files{i});
	plot(dat);
	title(files{i});
end
