CC = g++
CFLAGS = -Wall -Wextra -Wpedantic -std=c++14
LDFLAGS = -lm
SOURCES= main.cpp utils.cpp
OBJECTS=$(SOURCES:.cpp=.o)
BINARY = main

.PHONY: all
all: $(SOURCES) $(BINARY)

# todo make a nice makefile
# create a nice makefile which compiles all the files to a binary
# and cleans up the mess afterwards
$(BINARY): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f main *.o

	