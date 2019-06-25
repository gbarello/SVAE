#!/bin/bash

for loss in "gauss" "spikenslab"
do
    for ndim in 0 1 2
    do
	python run_sparse_vae.py $loss 1.5 --n_gauss_dim=$ndim --device=0
    done
done

