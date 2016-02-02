CC=icc
CFLAGS=-O3 -axAVX -std=gnu99 -openmp
LDFLAGS=
LIBS=-lm -openmp
all: lapl laplt laplv

lapl: lapl.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

laplt: laplt.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

laplv: laplv.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) lapl laplt laplv
