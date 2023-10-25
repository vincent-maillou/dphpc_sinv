CC = gcc
CFLAGS = -Wall -Wextra -pedantic -std=c99
LDFLAGS = -lm

SRCS = rgf_denseblocks.c
OBJS = $(SRCS:.c=.o)
EXEC = rgf_denseblocks

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
    $(CC) $(LDFLAGS) $^ -o $@

%.o: %.c
    $(CC) $(CFLAGS) -c $< -o $@

clean:
    $(RM) $(OBJS) $(EXEC)