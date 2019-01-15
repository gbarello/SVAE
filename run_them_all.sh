#!/bin/bash

python run_sparse_vae.py exp 1.5 --n_gauss_dim=0 --learning_rate=.001 --n_grad_step=200000 --device=2 --dataset=bruno

python run_sparse_vae.py exp 1.5 --n_gauss_dim=1 --learning_rate=.001 --n_grad_step=200000 --device=2 --dataset=bruno

python run_sparse_vae.py exp 1.5 --n_gauss_dim=2 --learning_rate=.001 --n_grad_step=200000 --device=2 --dataset=bruno
