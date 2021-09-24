CXX = gcc
CXX_FLAGS = -L.

test: test.c node.a
	$(CXX) $(CXX_FLAGS) -o $@ $< node.a
node.a: node.o
	ar rc $@ $<
	ranlib $@
%.o: %.c %.h
	$(CXX) $(CXX_FLAGS) -c $<
