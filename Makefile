CFLAGS=-Wall
LDFLAGS=
CC=g++
SRCDIR="./"
LIBDIR="./"

SOURCE = $(LIBDIR)mlp.cpp $(LIBDIR)matrix.cpp test.cpp

#OBJECTS=$(SOURCES:.c=.o)

all:
	$(CC) $(CFLAGS) $(SOURCE) -o mlp
