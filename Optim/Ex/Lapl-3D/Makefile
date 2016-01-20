CC=icc
CFLAGS=-O3 -axAVX -std=gnu99 -openmp
LDFLAGS=
LIBS=-lm -openmp

A: lapl.c
	$(CC) $(CFLAGS) $< -o lapl $(LDFLAGS) $(LIBS)

B: laplb.c
	$(CC) $(CFLAGS) $< -o lapl-blck $(LDFLAGS) $(LIBS)

C: laplv.c
	$(CC) $(CFLAGS) $< -o lapl-vect $(LDFLAGS) $(LIBS)

clean:
	$(RM) lapl lapl-blck lapl-vect
