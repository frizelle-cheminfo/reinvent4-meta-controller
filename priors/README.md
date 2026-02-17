# REINVENT4 Priors

This directory should contain REINVENT4 prior model files (`.prior` format).

## Download

Download the official REINVENT4 priors from Zenodo:

**https://zenodo.org/records/15641297**

## Installation

1. Download the priors from the link above
2. Extract the files
3. Place the `.prior` files in this directory

Example structure after installation:

```
priors/
├── README.md
├── reinvent.prior                    # Main prior model
└── libinvent.prior                   # LibINVENT prior (optional)
```

## Usage

The meta-controller will automatically detect priors in this directory. You can also specify custom prior paths in your configuration:

```yaml
arms:
  - name: "reinvent_qsar_explore"
    prior_path: "priors/reinvent.prior"
```

## Requirements

- REINVENT 4 installed
- Compatible PyTorch version (check Zenodo for requirements)

## More Information

- REINVENT4 Documentation: https://github.com/MolecularAI/REINVENT4
- Meta-Controller Documentation: See `docs/` directory
