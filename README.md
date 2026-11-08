# pinninit

pinninit is a PyTorch framework for physics-informed neural networks (PINNs).
A PINN is set up in a YAML parameter file, with its equations written as
Python functions, and trained on data saved to HDF5.

pinninit is in early development, so its API and parameter file format may
change.

## Installation

Create the conda environment and add pinninit to your Python path:

```bash
conda env create -f environment.yml
conda activate pinninit
export PYTHONPATH="/path/to/pinninit:$PYTHONPATH"
```

## Usage

```python
import torch
import pinninit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pinn = pinninit.PINN.from_yaml("param_file.yml", device)
pinn.train()
pinn.save("model.pth")
```

See `examples/` for usage in practice. Each example runs with `./run.sh` from
its directory. The MHD examples also need [Dedalus](https://dedalus-project.org)
to run their simulations.
