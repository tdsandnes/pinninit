1D MHD Alfven wave example

Reconstructs u and b over a whole 1D domain from measurements of b at one end.
Waves at two frequencies are driven in at x = 0 and reflect off x = 1, and the
PINN is trained on b at x = 1 for 20 <= t <= 30 together with the equations for
the Elsasser variables zp = u + b and zm = u - b.

Needs Dedalus (for simulate.py) and scipy, as well as pinninit.

Files:
    - run.sh:                           run the example with `./run.sh`
    - clean.sh:                         clean up files generated when running the example with `./clean.sh`
    - sim_config.py:                    The simulation parameters
    - simulate.py:                      Run the simulation with Dedalus and save snapshots of u and b
    - generate_data.py:                 Sample the training, validation and test sets from the snapshots
    - param_alfven_waves_1D.yml:        The parameter file for the PINN model trained in this example
    - alfven_waves_1D_equations.py:     The equations used in this example
    - train.py:                         Train and save PINN model
    - plot.py:                          Load model, print its test errors and plot output
