CXX := gcc
LDFLAGS := `pkg-config --static --libs glfw3 vulkan`

basic-triangle: basic-triangle.c
	$(CXX) $(LDFLAGS) -o $@ $^
