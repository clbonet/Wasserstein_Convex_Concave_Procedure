path_results="./results/cv_mmd"

wgf_gaussian() {
       	method=$1

        python xp_gaussian.py \
        --method $method \
        --n_try 25 \
        --path_results $path_results \
        --n_epochs 200000 \
        --target "gaussian" \
        --n_particles 500
}

wgf_mixture() {
       	method=$1

        python xp_gaussian.py \
        --method $method \
        --n_try 50 \
        --path_results $path_results \
        --target "gaussian_mixture" \
        --n_epochs 100000 \
        --n_particles 500
}



for method in "gd" "cccp_luu"  "cccp" "cccp_positive_decomposition"
do
  	wgf_gaussian $method
done


for method in "cccp" "gd" "cccp_luu" "cccp_positive_decomposition"
do
    wgf $method
done
