all: hw-orig hw-opt hw-orig-3 hw-opt-3

hw-orig:
	gcc 491hw3-template.c -o hw-orig

hw-opt:
	gcc -fopenmp -o hw-opt 521hw3.c

hw-orig-3:
	gcc -O3 491hw3-template.c -o hw-orig-3

hw-opt-3:
	gcc -O3 -fopenmp -o hw-opt-3 521hw3.c

.PHONY: clean
clean:
	rm hw-orig hw-opt hw-orig-3 hw-opt-3