CC=icc
CFLAGS=-O3 -axAVX -std=gnu99
LDFLAGS=
LIBS=-lm

all: mm-orig

mm-orig: mm.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

mm-blck: mm.c
	$(CC) $(CFLAGS) -DBLCK $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) mm-blck mm-orig
