# Citation and provenance

## Source magnetic-orderings dataset

The input data are taken unchanged from the `part1` compressed database snapshot in:

- Repository: https://github.com/ncfrey/magnetic-topological-materials
- Publication: Nathan C. Frey et al., “High-throughput search for magnetic and topological order in transition metal oxides,” *Science Advances* **6**, eabd1076 (2020).
- DOI: https://doi.org/10.1126/sciadv.abd1076

## ALIGNN

The structure model uses the `atomgptlab/alignn` pure-PyTorch interface specified by the notebook:

- Repository: https://github.com/atomgptlab/alignn
- Paper: K. Choudhary and B. DeCost, “Atomistic Line Graph Neural Network for improved materials property predictions,” *npj Computational Materials* **7**, 185 (2021).

## Spin-space symmetry

The final spin-space symmetry check uses `spinspg`:

- Package: https://pypi.org/project/spinspg/
- Reference: T. Shinohara et al., *Acta Crystallographica Section A* **80**, 94–103 (2024).
- DOI: https://doi.org/10.1107/S2053273323009257

Please cite the source database paper, ALIGNN paper, and spinspg reference in any paper that reports results from this workflow. Also cite this code repository with the version/commit used for the final run.
