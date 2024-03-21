# PerMFL (Personalized Multi-tier Federated Learning)

# Abstract: 
The key challenge of personalized federated learning (PerFL) is to capture the statistical heterogeneity properties of data with inexpensive communications and gain customized performance for participating devices. To address these, we introduced personalized federated learning in multi-tier architecture (\ourmodel{}) to obtain optimized and personalized local models when there are known team structures across devices. We provide theoretical guarantees of \ourmodel{}, which offers linear convergence rates for smooth strongly convex problems and sub-linear convergence rates for smooth non-convex problems. We conduct numerical experiments demonstrating the robust empirical performance of \ourmodel{}, outperforming the state-of-the-art in multiple personalized federated learning tasks.

Available datasets
1) MNIST 
2) FMNIST
3) EMNIST (10 classes and 62 classes)
4) Synthetic
5) CIFAR100

Available Algorithms
1) PerMFL (Ours)
2) pFedMe 
3) h-SGD
4) FedAvg
5) Per-FedAvg
6) Ditto
7) AL2GD

Requirements:
1. torch
2. torchvision
3. numpy
4. pandas
5. h5py
6. tqdm
