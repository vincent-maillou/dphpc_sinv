CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c99
LDFLAGS = -lm

all: rgf_denseblock

rgf_denseblock: rgf_denseblock.o utils.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

rgf_denseblock.o: rgf_denseblock.c utils.h
	$(CC) $(CFLAGS) -c $<

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f rgf_denseblock *.o

	