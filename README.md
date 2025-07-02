# Design of Conformational Biasing Mutations Using Inverse Folding

Code based on [ColabDesign/ProteinMPNN](https://github.com/sokrypton/ColabDesign) for designing mutations to bias protein conformational states. This repository is a work in progress. For more information, please see our preprint here: [https://www.biorxiv.org/content/10.1101/2025.05.03.652001v2.full](https://www.biorxiv.org/content/10.1101/2025.05.03.652001v2.full)

## Running CB

An example notebook for CB scoring of LplA is provided in the `examples` folder. You can also run the full workflow on Google Colab here:
<a target="_blank" href="https://colab.research.google.com/github/alicetinglab/ConformationalBiasing/blob/main/colab/CB.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Or easily for custom PDBs here:
<a target="_blank" href="https://colab.research.google.com/github/alicetinglab/ConformationalBiasing/blob/main/colab/CB_custom.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Requirements

Example code requires the following Python packages:

- `jax`
- `colabdesign`
- `scipy`
- `numpy`
- `pandas`

The complete environment used for running all code in the manuscript is provided in `requirements.txt`.
