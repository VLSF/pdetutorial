# Neural networks for PDEs: a tutorial

A collection of Jupyter notebooks on solving partial differential equations with neural networks, written in [JAX](https://github.com/jax-ml/jax). Each neural method is introduced next to a classical one, so that the two can be compared in accuracy and cost on the same problems.

The notebooks cover one-dimensional model problems:

- a **diffusion (elliptic) equation** $-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right)=f(x)$ with random high-contrast coefficients,
- the **viscous Burgers equation** $u_t+uu_x=\nu u_{xx}$ with random initial conditions,

and for each of them they explain the theory, show the implementation, generate a dataset, and report errors and timings. The operator learning notebooks then train neural operators (FNO, DeepONet, U-Net) on these datasets, and a final toy example on the periodic convection-diffusion equation $u_t+cu_x=\nu u_{xx}$ illustrates pretraining and finetuning, the mechanism behind foundation models.

## Structure

```
notebooks/
├── 01_classical_methods/
│   ├── 01_elliptic_1d_fem.ipynb
│   └── 02_burgers_1d_pseudospectral.ipynb
├── 02_pinns/
│   ├── 01_pinn_1d_elliptic.ipynb
│   └── 02_pinn_1d_burgers.ipynb
├── 03_operator_learning/
│   ├── 01_FNO.ipynb
│   ├── 02_DeepONet.ipynb
│   ├── 03_UNet.ipynb
│   └── 04_toy_foundation_model.ipynb
└── 04_hybrid_methods/      (coming soon)
```

The folder `04_hybrid_methods` is not populated yet and the notebooks for it will appear soon.

| Notebook | Contents |
| --- | --- |
| `01_classical_methods/01_elliptic_1d_fem.ipynb` | Finite element method with linear elements: weak form, stiffness and mass matrices, assembly. Generates a dataset of random diffusion problems. |
| `01_classical_methods/02_burgers_1d_pseudospectral.ipynb` | Fourier pseudospectral method with the FFT and integrating factor RK4 for the Burgers equation. Generates a dataset of random Burgers problems. |
| `02_pinns/01_pinn_1d_elliptic.ipynb` | Physics-informed neural networks (PINNs) with the Deep Ritz energy loss and a Siren network for the elliptic equation. Compares the solution time with FEM. |
| `02_pinns/02_pinn_1d_burgers.ipynb` | PINNs with the PDE residual loss for the Burgers equation on the space-time domain. Compares the solution time with the pseudospectral solver. |
| `03_operator_learning/01_FNO.ipynb` | Fourier Neural Operator (the factorised variant) trained on the Burgers dataset and on the two diffusion datasets. Reports the mean, median, minimal and maximal errors, the training curves, and the best and the worst test samples. |
| `03_operator_learning/02_DeepONet.ipynb` | DeepONet (a learned basis in the trunk network and coefficients from the projection of the input in the branch network) on the same three datasets, on a grid of 128 points. |
| `03_operator_learning/03_UNet.ipynb` | U-Net with residual blocks, group normalisation and skip connections on the same three datasets, on a grid of 128 points. |
| `03_operator_learning/04_toy_foundation_model.ipynb` | A toy foundation model: a small FNO is pretrained on convection and on diffusion, and then applied zero-shot and finetuned on the convection-diffusion problem, compared with the training from scratch (data efficiency for 5 to 3000 samples). The data are generated in the notebook. |

## Running the notebooks

The notebooks in `02_pinns` and in `03_operator_learning` (except `04_toy_foundation_model.ipynb`, which generates its own data) read the datasets produced by the notebooks in `01_classical_methods`, which save them to the folder `data` in the root of the repository (the folder is created on the first run). Run the notebooks in `01_classical_methods` first. The notebooks save their figures to the same folder, which is not tracked by git.

The operator learning notebooks train for 30 s to a few minutes each on a GPU (the experiments in `04_toy_foundation_model.ipynb` take about 35 minutes in total, mostly for the selection of hyperparameters and the runs with many seeds).

## Dependencies

- Python 3
- [JAX](https://github.com/jax-ml/jax) (a GPU is recommended for the PINN and the operator learning notebooks, but not required)
- [Equinox](https://github.com/patrick-kidger/equinox) for the neural networks
- [Optax](https://github.com/google-deepmind/optax) for the optimisers and learning rate schedules
- [NumPy](https://numpy.org) and [Matplotlib](https://matplotlib.org)
- [Jupyter](https://jupyter.org)

The notebooks were run with:

| Package | Version |
| --- | --- |
| Python | 3.9 |
| jax, jaxlib | 0.4.30 |
| equinox | 0.11.10 |
| optax | 0.2.4 |
| numpy | 2.0.2 |
| matplotlib | 3.9.4 |

Installation, for example:

```
pip install jax equinox optax numpy matplotlib jupyter
```

For a GPU install JAX as described in the [JAX documentation](https://docs.jax.dev/en/latest/installation.html).
