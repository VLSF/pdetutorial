# Neural networks for PDEs: a tutorial

A collection of Jupyter notebooks on solving partial differential equations with neural networks, written in [JAX](https://github.com/jax-ml/jax). Each neural method is introduced next to a classical one, so that the two can be compared in accuracy and cost on the same problems.

The notebooks cover one-dimensional model problems:

- a **diffusion (elliptic) equation** $-\frac{d}{dx}\left(a(x)\frac{du}{dx}\right)=f(x)$ with random high-contrast coefficients,
- the **viscous Burgers equation** $u_t+uu_x=\nu u_{xx}$ with random initial conditions,

and for each of them they explain the theory, show the implementation, generate a dataset, and report errors and timings.

## Structure

```
notebooks/
├── 01_classical_methods/
│   ├── 01_1d_elliptic_fem.ipynb
│   └── 02_burgers_1d_pseudospectral.ipynb
├── 02_pinns/
│   ├── 01_pinn_1d_elliptic.ipynb
│   └── 02_pinn_1d_burgers.ipynb
├── 03_operator_learning/   (coming soon)
└── 04_hybrid_methods/      (coming soon)
```

The folders `03_operator_learning` and `04_hybrid_methods` are not populated yet and the notebooks for them will appear soon.

| Notebook | Contents |
| --- | --- |
| `01_classical_methods/01_1d_elliptic_fem.ipynb` | Finite element method with linear elements: weak form, stiffness and mass matrices, assembly. Generates a dataset of random diffusion problems. |
| `01_classical_methods/02_burgers_1d_pseudospectral.ipynb` | Fourier pseudospectral method with the FFT and integrating factor RK4 for the Burgers equation. Generates a dataset of random Burgers problems. |
| `02_pinns/01_pinn_1d_elliptic.ipynb` | Physics-informed neural networks (PINNs) with the Deep Ritz energy loss and a Siren network for the elliptic equation. Compares the solution time with FEM. |
| `02_pinns/02_pinn_1d_burgers.ipynb` | PINNs with the PDE residual loss for the Burgers equation on the space-time domain. Compares the solution time with the pseudospectral solver. |

## Running the notebooks

The notebooks in `02_pinns` read the datasets produced by the notebooks in `01_classical_methods`, which save them to the folder `data` in the root of the repository (the folder is created on the first run). Run the notebooks in `01_classical_methods` first.

## Dependencies

- Python 3
- [JAX](https://github.com/jax-ml/jax) (a GPU is recommended for the PINN notebooks, but not required)
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
