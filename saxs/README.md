# SAXS Analysis Files

This folder contains three example notebooks for processing SAXS data:

- `trim_files.ipynb` trims raw data files, removing low quality early data points and thresholding at a max Q value
- `oligomer.ipynb` runs [Oligomer](https://www.embl-hamburg.de/biosaxs/manuals/oligomer.html) analysis (part of the ATSAS software package) on trimmed data files
- `denss.ipynb` runs [DENSS](https://tdgrant.com/about-denss/) analysis on trimmed data files.

We also provides two scripts for analysis of outputs in the `analysis` folder, as well as exact PDB models used for oligomer analysis in the `pdbs` folder.

If you use DENSS/Oligomer, please cite the following articles in your work:

- Grant TD. **Ab initio electron density determination directly from solution scattering data.** _Nature Methods_ (2018). doi: [10.1038/nmeth.4581](https://www.nature.com/articles/nmeth.4581)
- Konarev PV, Volkov VV, Sokolova AV, Koch MHJ, Svergun DI. **PRIMUS: a Windows PC-based system for small-angle scattering data analysis.** _Journal of Applied Crystallography_ (2003). doi: [10.1107/S0021889803012779](https://journals.iucr.org/paper?S0021889803012779)
