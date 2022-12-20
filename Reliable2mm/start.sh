ulfm_image=abouteiller/mpi-ft-ulfm
function mpirun {
    docker run --user $(id -u):$(id -g) --cap-drop=all --security-opt label:disabled -v $PWD:/sandbox $ulfm_image mpirun --map-by :oversubscribe --mca btl tcp,self $@
}
function mpicc {
    docker run --user $(id -u):$(id -g) --cap-drop=all --security-opt label:disabled -v $PWD:/sandbox $ulfm_image mpicc $@
}

mpicc tmp.c
mpirun --with-ft ulfm -n $1 ./a.out