% Let G = (V,E) be an undirected graph
% Goal is to partition V into disjoint subsets A and B.
% Let W(i,j) = w_ij be a symmetric matrix, where w_ij is the edge weight
%  connecting vertex i with vertex j.
% Let D(i,i) = d_i, where d_i = sum_j w(i,j).
% Will solve Ncut(x), where x is an indicator vector. x(i) = 1 indicates node i
%  is in subset A, else x(i) = -1. (Ideally) May need to choose splitting point.
% Solve eigensystem D^(1/2) (D - W) D^(1/2) z = lambda z, and use second
%  smallest eigenvalue/vector as partition.

pkg load statistics;

fig = 1;

% Points to use
points = [
	0,1;
	0,1.1;
	0.1,1;

	1,0;
	1.1,0;
	1,-0.1;

	-1,0;
	-1.1,0;
	-1,0.1;
];

N = size(points,1);

figure(fig++);
clf();
plot(points(:,1), points(:,2),'o')

% Construct graph using kernel function for similarity
% sigma is neighborhood size parameter
sigma2 = 0.9;
kern = @(xi,xj)(exp(-norm(xi-xj, 2, 'rows').^2 ./ (2 * sigma2)));

G = [];
for k = [1:N]
	G(k,:) = kern(points(k,:), points);
end

% Heatmap for own curiosity
figure(fig++);
clf();
imshow(G);

% Prepare for Ncut
d = sum(G, 2);
D = diag(d);

% Compute product
M = D^0.5 * (D - G) * D^0.5;

% Get eigenvalue(s)
[EVE, EVA] = eigs(M);
