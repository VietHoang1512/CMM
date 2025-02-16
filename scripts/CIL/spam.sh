for alpha_merge in .4 .3 .5  .6 ; do
    for n_splits in 5 10 20  ; do
    sbatch scripts/CIL/all.sh $alpha_merge $n_splits
    done
done