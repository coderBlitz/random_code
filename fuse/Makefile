CC := gcc
CFLAGS := $$(pkg-config --cflags fuse)

LIBS := $$(pkg-config --libs fuse)

%: %.c
	$(CC) $(CFLAGS) $(LIBS) $< -o $@
