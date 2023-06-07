CC 			=	gcc
CFLAGS		=	-O3 -Wunused-result
PROG		=	symgs/symgs-test spmv/spmv-test smith-waterman/sw-test

all:$(PROG)

spmv/spmv-test: spmv/spmv-csr.c
	$(CC) $(CFLAGS) $^ -o $@ 

.PHONY:clean
clean:
	rm -f $(PROG)
