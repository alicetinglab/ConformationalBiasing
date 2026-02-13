<img src="https://raw.githubusercontent.com/alicetinglab/ConformationalBiasing/main/images/CB_cartoon.jpg" align="center" alt="drawing" width="1000">

This repository accompanies our manuscript: [Computational design of conformation-biasing mutations to alter protein functions](https://www.science.org/doi/10.1126/science.adv7953).

In this repository, we implement a computational pipeline for designing protein variants with altered conformational state preferences, based on using inverse-folding models to score a protein across known alternative conformations. We include a version that runs on Google Colab, as well as multiple individual notebooks for running example predictions for _E. coli_ lipoic acid ligase (LplA) across a variety of inverse-folding models.

## Method Overview

CB takes two protein structures representing different conformational states of a protein as input and:

1. Aligns the two protein sequences across the structures to generate a consensus sequence
2. Generates mutant sequences (by default, all single mutants) on the consensus sequence
3. Maps those variant sequences to each structure, and scores on both structures to calculate bias predictions

## Models and Code

We implement CB with four different inverse folding models:

- **ProteinMPNN** (Default): [Reference](https://www.science.org/doi/10.1126/science.add2187), [Code](https://github.com/dauparas/ProteinMPNN)
- **Frame2Seq**: [Reference](https://www.biorxiv.org/content/10.1101/2023.12.15.571823v1), [Code](https://github.com/dakpinaroglu/Frame2seq)
- **ThermoMPNN**: [Reference](https://www.pnas.org/doi/10.1073/pnas.2314853121), [Code](https://github.com/Kuhlman-Lab/ThermoMPNN)
- **ESM-IF1**: [Reference](https://proceedings.mlr.press/v162/hsu22a/hsu22a.pdf), [Code](https://github.com/facebookresearch/esm)

Pipeline prioritizes running ProteinMPNN, as it is the most extensively validated in our manuscript. Additional models can be run at the users preference.

Model codebases were modified slightly in order to ensure compatability and ease of use. No weights were modified, nor any other functional changes. Running ProteinMPNN is enabled by code from [ColabDesign](https://github.com/sokrypton/ColabDesign) (thank you to [solab](https://www.solab.org/)!). Forked versions of the other repositories can be found at the following links: [Frame2Seq](https://github.com/andrewxue98/Frame2seq), [ThermoMPNN](https://github.com/andrewxue98/ThermoMPNN), and [ESM-IF1](https://github.com/andrewxue98/esm)

## Running CB

CB is designed to be a lightweight method that can be run on limited compute across thousands of protein variants. As such, we highly recommend running CB using our provided notebook on Google Colab, as the run time on free GPU-enabled instances is very reasonable. The only required inputs are two structures of the same protein in different conformations (see FAQ below).

<a target="_blank" href="https://colab.research.google.com/github/alicetinglab/ConformationalBiasing/blob/main/colab/CB.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Running CB locally - environment setup

For more advanced use, we provide example notebooks for usage of each model in the `examples` folder. We recommend setting up a new conda environment using the following steps.

```
conda create -n cb python=3.11 jupyter

conda install pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric biotite gemmi

#install versions of each model that have been modified to be compatible
pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
pip install git+https://github.com/andrewxue98/esm.git
pip install git+https://github.com/andrewxue98/ThermoMPNN.git
pip install git+https://github.com/andrewxue98/Frame2seq.git
```

You can also use the provided `environment.yml` to create this environment.

## FAQ

- What structures should I use?
  - Structures should be for the same protein (not orthologs, ideally minimal point mutations) and should have overlapping coverage of the protein backbone. They should ideally be similar in resolution as well.
- How different do the conformational states need to be?
  - We observe that we're able to find mutants that are predicted to be state-biased even in situations where the conformational change is subtle between states (e.g. FabZ). Our recommendation is to run CB and take a look at the results. If state 1 and state 2 scores are very highly correlated, your structures were likely too similar.
- Are AlphaFold/predicted structures acceptable?
  - Yes, in our experience, if the predictions are high confidence. If possible, use experimentally solved structures as predicted structures may add some noise to predictions.
- How big can my protein be? How many variants can I score?
  - We are able to easily score thousands of variants on proteins of length >1000AA. Runtime should scale linearly with number of variants scored, and exponentially with protein length. In general, compute should not be limiting for CB analysis except in extreme cases.
- Does CB work only for single mutants?
  - We think that CB is applicable variants that are further than one mutation away from wild-type. For example, in our K-Ras dataset, we are able to see effects on a population of designed double mutants.
- Which IF model performs the best for CB?
  - ProteinMPNN seems to perform consistently well across all of our datasets, which is why we use it by default. In limited comparisons, other models seem to sometimes perform worse or better, depending on context. We observe that biasing mutations predicted by agreement between models seem to have the strongest effects.

## Issues, Troubleshooting, and Limitations

**Issues:**

1. Please first read through the troubleshooting guide below and see if any of those steps are able to resolve your issues.
2. If that fails, please open a Github and include detailed information about the issue you encounter and your inputs.
3. For any non-breaking bugs you encounter, please open an issue as well (thanks!).

**Troubleshooting:**

1. Check if you have these common issues:
   - _Uploaded duplicate structures or extremely similar structures:_
     - CB scatter plot will show points with almost perfect correlation.
     - Please ensure you're uploading two different conformational states of a protein. We see good results even on subtle shifts so if points are almost perfectly correlated, it's likely that the conformational difference captured in the structure is not meaningful.
   - _Uploaded structures of different protein:_
     - Very poor alignment on alignment chart, almost no mutant sequences generated.
     - Please check your structures in the structure viewer below.
   - _Uploaded structures of protein orthologs:_
     - Poor/spotty alignment on alignment chart, many points in alignment missing, fewer mutant sequences generated
     - Upload a structure with a matching sequence (could be AlphaFold generated using original structure as a template)
   - _Poor overlap of resolved backbones:_
     - Would show as non-overlapping alignments on alignment chart
     - Some structures are only solved on part of the backbone. This pipeline only works on regions where the structure is solved in both conformations. Please generate/find structures where the backbone overlap is good.
   - _Selected incorrect chains:_
     - Possibly poor alignment on alignment chart, fewer mutant sequences generated.
     - Please check chains in structure viewer below and make sure you're selecting the correct chain
2. Check structure files in pyMOL or similar:
   - Ensure chains are not named abnormally
   - Check for unnatural amino acids (some are handled by default, but not all)
   - Check for missing atoms
   - Sometimes [PDBFixer](https://github.com/openmm/pdbfixer) can help resolve these issues

**Current Limitations:**

- ESM-IF1: runtime is orders of magnitude slower than other models
- ThermoMPNN: currently only setup to handle single mutants
- Colab notebook and example scripts support single chain scoring only
- No support of non-canonical amino acids except for conversion to canonical
- Sensitivity across all variants: we think that our predicted-biased variants have a good hit-rate (see LplA data from manuscript). However, we have not exhaustively characterized false-negatives; it is likely that many mutations we predict as "neutral" could have some effect on conformational occupancy.

## Other repository contents

- In `saxs` we include code for SAXS data processing with DENSS and Oligomer, as well as downstream analysis.
- In `modeling` we include a script for ODE modeling of LplA reaction rates under conformational biasing conditions.
- In `pdbs` we include organized PDBs that were used in the standard CB workflow for all of results shown in our manuscript. More detailed description of the files is in `pdbs\annotation.md`.

## Citing this work

For the CB method:

- Cavanagh PE, Xue AG, Dai S, Qiang A, Matsui T, Ting AY. **Computational design of conformation-biasing mutations to alter protein functions.** _Science_ (2026). doi: [10.1126/science.adv79531](https://doi.org/10.1126/science.adv7953)

If you use the following models, please cite them as well:

**ProteinMPNN**

- Dauparas J, Anishchenko I, Bennett N, Bai H, Ragotte RJ, Milles LF, _et al._ **Robust deep learning–based protein sequence design using ProteinMPNN.** _Science_ (2022). doi: [10.1126/science.add2187](https://doi.org/10.1126/science.add2187)

**Frame2seq**

- Akpinaroglu D, Seki K, Guo A, Zhu E, Kelly MJS, Kortemme T. **Structure-conditioned masked language models for protein sequence design generalize beyond the native sequence space** _bioRxiv_ (2023).

**ThermoMPNN**

- Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B. **Transfer learning to leverage larger datasets for improved prediction of protein stability changes.** _Proceedings of the National Academy of Sciences (PNAS)_ (2024). doi: [10.1073/pnas.2314853121](https://doi.org/10.1073/pnas.2314853121)

**ESM-IF1**

- Hsu C, Verkuil R, Liu J, Lin Z, Hie B, Sercu T, Lerer A, Rives A. **Learning inverse folding from millions of predicted structures.** _Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162_ (2022): 8946–8970.
