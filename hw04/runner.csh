#!/usr/bin/env csh

source /var/mpi-selector/data/openmpi-1.7.2.csh
make

set i = 1
set j = 1
while ( $i < 5 )
	repeat 3 mpirun -H borg,granville,lamarr,perlman -npernode $i ./heat
	@ i++
end