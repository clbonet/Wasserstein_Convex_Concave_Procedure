# Difference of Convex Programming in the Wasserstein Space with Applications to MMD Optimization

This repository contains the code to reproduce the experiments [Difference of Convex Programming in the Wasserstein Space with Applications to MMD Optimization](https://arxiv.org/abs/2606.27767). In this paper, we lift the convex-concave procedure (CCCP) to the Wasserstein space to minimize functionals which admit difference-of-convex decomposition. In particular, we focus on the Maximum Mean Discrepancy, for which we develop explicit DC decompositions.


## Abstract

Optimizing functionals over the space of probability measures is now ubiquitous in machine learning. A widely used approach is to perform the optimization directly over the Wasserstein space, but many objective functionals of practical interest are non-convex along Wasserstein geodesics, making the analysis of standard first-order methods challenging. In this work, we study a class of objectives over the Wasserstein space that admit a difference-of-convex (DC) decomposition  and we lift the classical convex-concave procedure (CCCP) to this setting. Under smoothness and strong convexity assumptions on the convex components of the decomposition, we prove almost stationarity along the iterates of the resulting algorithm. Our main focus is on the Maximum Mean Discrepancy (MMD) and the Energy Distance (ED) functionals, for which we develop explicit Wasserstein DC decompositions, and establish local convergence of the scheme under mild assumptions. Empirically, we show that well-chosen DC decompositions yield faster and more stable convergence than Wasserstein gradient descent on these MMD objectives.


## Citation

```
@article{bonet2025difference,
    title={{Difference of Convex Programming in the Wasserstein Space with Applications to MMD Optimization}},
    author={Clément Bonet and Pierre-Cyril Aubin-Frankowski and Youssef Mroueh},
    year={2026},
    journal={arXiv preprint arXiv:2606.27767}
}
```


## Experiments

- Experiments with the Energy Distance (e.g. Figure 1 and 2) can be reproduced by running the code from the folder `xp_Energy_Distance`
- Experiments with the MMD with Gaussian kernel (e.g. Figure 3) can be reproduced by running the code from the folder `xp_MMD`
