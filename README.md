# Equilibrium-Informed Neural Networks (EINNs)

## Overview

This repository contains the Python implementation of the Equilibrium-Informed Neural Networks (EINNs) approach proposed in the manuscript:

**Machine Learning for Complex Systems Dynamics: Detecting Bifurcations in Dynamical Systems with Deep Neural Networks**

The implementation demonstrates how deep neural networks can be used to estimate equilibrium points and detect saddle-node bifurcation thresholds in nonlinear dynamical systems.

## Repository Organization

The repository contains the following files and directories:

| File/Directory | Description |
|---|---|
| `Example_1.py` | EINNs implementation of the first single-equation ecological model (equation (4)). |
| `Example_2.py` | EINNs implementation of the second ecological model (equation (6)). |
| `Example_3.py` | EINNs implementation of the coupled two-equation Alzheimer’s disease model (equation (10)). |
| `Example_4.py` | Implementation of the extended system of equations (equation (13)). |

## Network Architecture

The EINNs implementation uses deep feed-forward neural networks with hyperbolic tangent activation functions. The network is optimized with the Adam optimizer, and the loss function is formulated in terms of residuals of the governing equilibrium equations.

The specific network architecture, sampling strategy, and optimization parameters used for each numerical example are described in the corresponding source files and in the manuscript.


## Execution

The numerical examples can be executed individually using:

```bash
python Example_1.py
python Example_2.py
python Example_3.py
python Example_4.py
```

## Reproducibility

Each Python script contains the model equations, network configuration, training procedure, sampling strategy, and plotting routines required to reproduce the corresponding numerical results.

The generated figures can be compared with the equilibrium and bifurcation diagrams presented in the manuscript.

## Contact

For questions concerning the implementation or the EINNs methodology, please contact the corresponding authors.
