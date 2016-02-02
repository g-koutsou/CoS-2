CC=icc
CFLAGS=-O3 -axAVX -std=gnu99
LDFLAGS=
LIBS=-lm
all: space-time-soa space-time-aosoa

space-time-soa: space-time-soa.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

space-time-aosoa: space-time-aosoa.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) space-time-soa space-time-aosoa
