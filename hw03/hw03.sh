#!/usr/bin/env bash

NUM_THREADS=8
DEBUG=false
MAKE=true

USAGE="To run: ./hw03 -t <threads> -d (display debug info) -m (do not remake project)\n\
If no arguments are supplied: threads = 8, remake project, no debug info"

while getopts "t:md?" opt; do
	case $opt in
		t)
			NUM_THREADS=${OPTARG}
			;;
		d)
			DEBUG=true
			;;
		m)
			MAKE=false
			;;
		\?)
			echo -e ${USAGE}
			exit
			;;
	esac
done

# clean out old files and compile the programs
if [ ${MAKE} = true ]; then
	if [ ${DEBUG} = false ]; then
		make clean > /dev/null
		make > /dev/null
	else
		make clean
		make
	fi
fi

# set the number of threads for OpenMP
export OMP_NUM_THREADS=${NUM_THREADS}

echo "Unoptimized Results:"
for i in `seq 1 3`; do
	./hw-orig
done

echo " "

echo "Optimized Results:"
THREADS="1 2 4 8 16"
for t in ${THREADS}; do
	# set the number of threads for OpenMP
	export OMP_NUM_THREADS=${t}
	echo "${t} Threads"
	for i in `seq 1 3`; do
		./hw-opt
	done
done

echo "Compiler Optimized Results:"
for t in ${THREADS}; do
	# set the number of threads for OpenMP
	export OMP_NUM_THREADS=${t}
	echo "${t} Threads"
	for i in `seq 1 3`; do
		./hw-opt-3
	done
done
