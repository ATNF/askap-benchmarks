cmpref=cray
for cmp in {gnu,cray,aocc}
do
    module swap PrgEnv-${cmpref} PrgEnv-${cmp}
    echo "============= ${cmp} ==============="
    module list 

    #make clean
    #make CXX=CC MPIFLAGS=-D_MPI MPICXX=CC
    ./build_cpu.sh 
    mkdir -p lib-${cmp}
    mv lib/lib* lib-${cmp}/
    echo "Done \n "
    echo "============= ${cmp} ==============="
    module swap PrgEnv-${cmp} PrgEnv-${cmpref}
done

