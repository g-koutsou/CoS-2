CC=icc
CFLAGS=-O3 -axAVX -std=gnu99
LDFLAGS=
LIBS=-lm
all: space-time-aos

space-time-aos: space-time-aos.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

space-time-soa: space-time-soa.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

space-time-soa-r: space-time-soa-r.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

space-time-aosoa-r: space-time-aosoa-r.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)


clean:
	$(RM) space-time-soa space-time-aos space-time-aosoa-r space-time-soa-r
