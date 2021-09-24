function xn = newtons(f, F, xinit, tol, max_iter);
	xn = xinit;
	iter = 1;
	err = tol+1;
	while iter <= max_iter && err > tol;
		dn = F(xn)\f(xn);
		xn = xn - dn;

		err = norm(f(xn));
		iter = iter+1;
	endwhile

	if iter == max_iter;
		disp("Max iterations reached");
	endif
end
