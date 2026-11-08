# 1D MHD Alfven wave example

Reconstructs u and b over a whole 1D domain from measurements of b at one end.
Waves at two frequencies are driven in at x = 0 and reflect off x = 1, and the
PINN is trained on b at x = 1 for 20 <= t <= 30 together with the equations for
the Elsasser variables zp = u + b and zm = u - b.

Needs Dedalus (for simulate.py), scipy and colorcet, as well as pinninit.

Files:

- `run.sh`: run the example with `./run.sh`
- `clean.sh`: clean up files generated when running the example with `./clean.sh`
- `sim_config.py`: the simulation parameters
- `simulate.py`: run the simulation with Dedalus and save snapshots of u and b
- `generate_data.py`: sample the training, validation and test sets from the snapshots
- `param_alfven_waves_1D.yml`: the parameter file for the PINN model trained in this example
- `alfven_waves_1D_equations.py`: the equations used in this example
- `train.py`: train and save PINN model
- `plot_losses.py`: plot the training and validation losses
- `plot_snapshots.py`: plot the simulation's and the PINN's u and b at a few times
- `plot_error_grid.py`: print the test errors and plot the absolute error over (x, t)
- `paper.mplstyle`: the plotting style
