1D MHD example from Fournier et al. (2007)

Reconstructs u and b over the whole domain -1 <= x <= 1 and 0 <= t <= 0.2 from
measurements of b. The PINN is trained on b on a grid of points inside the
domain and the momentum and induction equations. The boundary conditions on u
and b at x = -1 and x = 1 hold by construction (hard boundary conditions).

Needs Dedalus (for simulate.py), scipy and colorcet, as well as pinninit.

Files:
    - run.sh:                           run the example with `./run.sh`
    - clean.sh:                         clean up files generated when running the example with `./clean.sh`
    - sim_config.py:                    The simulation parameters
    - simulate.py:                      Run the simulation with Dedalus and save snapshots of u and b
    - generate_data.py:                 Sample the training, validation and test sets from the snapshots
    - param_fournier07_1D.yml:          The parameter file for the PINN model trained in this example
    - fournier07_1D_equations.py:       The equations used in this example
    - train.py:                         Train and save PINN model
    - plot_losses.py:                   Plot the training and validation losses
    - plot_snapshots.py:                Plot the simulation's and the PINN's u and b at a few times
    - plot_error_grid.py:               Print the test errors and plot the absolute error over (x, t)
    - paper.mplstyle:                   The plotting style
