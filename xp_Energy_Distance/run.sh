xp_riesz() {
       	method=$1
        target=$2

        python xp_shapes \
        --method $method \
        --n_try 100 \
        --path_results $path_results \
        --n_epochs 5000 \
        --target $target \
        --n_particles 500
}


for target in "cat" "disk" "heart" "spiral" "gaussian" "gaussian_mixture"
do
  for method in "gd" "cccp" "fb"
  do
    xp_riesz $method $target
  done
done



wgf_cifar() {
       	method=$1
        epochs=$2

        python xp_imgs.py \
        --method $method \
        --path_results $path_results \
        --n_epochs $epochs \
        --n_data_by_class 50 \
        --path_data $path_data \
        --tgt_dataset "CIFAR10" \
        --save_interval 1000 \
        --n_try 5
}

wgf_cifar "gd" 200000
wgf_cifar "cccp" 40000
