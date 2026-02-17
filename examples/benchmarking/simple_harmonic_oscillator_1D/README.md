Note to self: example works on commit 387662632d01b41f873ae497209f36ffdf03f99d

Simple harmonic oscillator example

Finds solution by constraining loss at data points and the residual of f(x)'' + f(x) = 0.

Files:
    - run.sh:                                       run the example with `./run.sh`
    - clean.sh:                                     clean up files generated when running the example with `./clean.sh`
    - param_simple_harmonic_oscillator_1D.yml:      The parameter file for the PINN model trained in this example
    - simple_harmonic_oscillator_1D_equations.py:   The equations used in this example
    - train.py:                                     Train and save PINN model
    - plot.py:                                      Load model and plot output
    - datasets.py and generate_data.py:             Used to save data in correct format. The way this is done is temporary,
                                                    since it would be nicer to use the data structure that is used internally
                                                    in the code e.g. Coords, Fields classes.
