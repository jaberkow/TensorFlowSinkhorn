# Description

This repository contains an implementation of the sinkhorn algorithm (1) in TensorFlow so that it can differentiated through.

## Contents

* `tf_wasserstein.py` contains the necessary tensorflow functions, notably the function `sinkhorn_loss` that computes the sinkhorn distance
* `swiss_roll_demo.ipynb` contains an example use of the sinkhorn_loss, implementing a sinkhorn autoencoder (2) on the swiss roll dataset

# Requirements

* `tf_wasserstein.py` requires TensorFlow 1.1 or greater and all dependencies therein
* `swiss_roll_demo.ipynb`uses [TensorFlow 2](https://www.tensorflow.org/alpha/guide/effective_tf2) and matplotlib

# Further Reading

1. Cuturi, Marco. "Sinkhorn distances: Lightspeed computation of optimal transport." *Advances in neural information processing systems.* (2013). http://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport
2. Patrini, Giorgio, et al. "Sinkhorn autoencoders." *arXiv preprint arXiv:1810.01118* (2018). https://arxiv.org/abs/1810.01118
