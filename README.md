# Integral-Renyi-ELBO
Implements the Integral Renyi ELBO from our paper https://arxiv.org/abs/1807.01889

The _"Tensorflow_py_func_with_grad.py"_ is forked from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342 ; 
it implements a simple example on how to use custom gradients in tensorflow


I built on this example to create my own custom gradients, for the special Dawson math function,
which I needed in the computation of the Integral Renyi ELBO from our paper 
https://arxiv.org/abs/1807.01889

In *"fixed_Erf_Dawson_my_custom_gradient.py"*, I use the method given in "Tensorflow_py_func_with_grad.py", to implement a custom gradient in tensorflow, for the numpy functions 'erf' and 'dawsn'. Without these custom gradients, if the cost function contains numpy functions wrapped in tensorflow, the VAE cannot perform backpropagation for some of the training variables involved in this cost function. One needs to compute the gradients of these numpy functions by hand, wrap them in the tensorflow graph and give them explicitly to the gradient node, to perform backpropagation on them.

The script *"fixed_simpleVAE_kl_renyi_cdf.py"* implements the integral ELBO (IELBO) bound, derived using the Kullback-Leibler divergence.

The script *"fixed_simpleVAE_renyi_cdf.py"* implements the integral Renyi lower bound (IRELBO) bound, derived using the Renyi divergence.
