Simple harmonic oscillator example

Finds solution by constraining loss at data points and the residual of f(x)'' + f(x) = 0.

Needs matplotlib (for plot.py), as well as pinninit.

Files:
    - run.sh:                                       run the example with `./run.sh`
    - clean.sh:                                     clean up files generated when running the example with `./clean.sh`
    - generate_data.py:                             Generate and save the training, validation and test sets
    - param_simple_harmonic_oscillator_1D.yml:      The parameter file for the PINN model trained in this example
    - simple_harmonic_oscillator_1D_equations.py:   The equations used in this example
    - train.py:                                     Train and save PINN model
    - plot.py:                                      Load model and plot output
