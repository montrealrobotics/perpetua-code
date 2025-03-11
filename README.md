# Perpetua

This is the code of the paper ["Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments"](https://montrealrobotics.ca/perpetua). We provide estimation and learning examples for both Perpetua and our mixture models.

## Installation

First clone the repository:

```bash
git clone git@github.com:montrealrobotics/perpetua-code.git
```

Then create a virtual environment and run the following commands.

```bash
# Install Venv
pip3 install --upgrade pip
pip3 install --user virtualenv
# Create Env
cd perpetua-code
python3 -m venv venv
# Activate env
source venv/bin/activate
```

Continue installing the project dependencies.

```bash
pip3 install -r requirements.txt
pip3 install -e .
export PYTHONPATH=~/<CODE_PATH>/perpetua-code:$PYTHONPATH
```

Optional: we found that using [JAX](https://docs.jax.dev/en/latest/installation.html) in CPU makes our code run faster. Therefore, we recommend running the following command to attain the fastest performance.

```bash
export JAX_PLATFORM_NAME=cpu
```

## Getting Started

We provide several examples on how to use Perpetua and its individual mixture components. All these examples are located in the `examples` folder.

### Running the Mixture of Persistence and Emergence Filters

To run the mixture of persistence filters, run the following command

```bash
python3 examples/mixture_persistence.py --prior lognorm
```

Similarly, to run the mixture of emergence filters use
```bash
python3 examples/mixture_emergence.py --prior lognorm
```

The previous example runs with three different choices of prior: `exponential`, `lognorm` and `weibull`.

> All the resulting images/artifacts are stored in the `assets` folder.

### Running Perpetua

Perpetua can be run using different choices of prior

```bash
python3 examples/perpetua.py --prior lognorm --num_steps 10 --eps 0.1
```

`eps` is the mixing coefficient after a reset (see Eq. (18)) and `num_steps` are the number of samples used during simulation. Larger values for `num_steps` makes the results more precise but makes Perpetua slower.

### Running Learning Examples

For training a mixture of persistence filters using a mixture of exponential priors run this.

```bash
python3 examples/mixture_learning_exponential.py --filter persistence --n_seq 20
```

By changing `--filter emergence` the same script will train a mixture of emergence filters instead. The parameter `n_seq` represents the number of training sequences used during training.

In a similar way we can train mixtures of persistence filters using a mixture of log-normal priors.

```bash
python3 examples/mixture_learning_exponential.py --filter persistence --n_seq 20
```

For further details, see Section IV-D in the paper and our supplemenary material linked [here](https://montrealrobotics.ca/perpetua)

### Room Environment Example

We provide a full example in a short sequence of the room environment where we use the [ruptures](https://centre-borelli.github.io/ruptures-docs/) library to obtain the training sequences. Both train and test sequences last for three hours. To run the full example, use the following command.

```bash
python3 examples/room.py --prior lognorm
```

where available choices of prior are `lognorm` and `exponential`.

## Citation

If you found this code useful, please use the following citation.

```bibtex
@article{saavedra2025perpetua,
	title        = {Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments},
	author       = {Saavedra-Ruiz, Miguel and Nashed, Samer and Gauthier, Charlie and Paull, Liam},
	year         = 2025,
	journal      = {arXiv preprint},
}  
```



