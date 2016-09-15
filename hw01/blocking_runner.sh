#!/usr/bin/env bash

make clean

make

run ()
{
	for i in `seq 1 3`;
	do
		./hw01ryfcs491 $1 $2 $3 $4 >> blocking_results.txt
	done
}

IT=1
JT=1
KT=1
LT=1
NT=128

while [ $IT -lt $NT ];
do
	while [ $JT -lt $NT ];
	do
		while [ $KT -lt $NT ];
		do
			while [ $LT -lt $NT ];
			do
				run $IT $JT $KT $LT
				LT=$((LT*2))
			done
			KT=$((KT*2))
		done
		JT=$((JT*2))
	done
	IT=$((IT*2))
done
