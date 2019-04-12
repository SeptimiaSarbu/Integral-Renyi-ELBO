# Integral-Renyi-ELBO
Implements the Integral Renyi ELBO from our paper https://arxiv.org/abs/1807.01889

The "Tensorflow_py_func_with_grad.py" is forked from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342 ; 
it implements a simple example on how to use custom gradients in tensorflow


I built on this example to create my own custom gradients, for the special Dawson math function,
which I needed in the computation of the Integral Renyi ELBO from our paper 
https://arxiv.org/abs/1807.01889
