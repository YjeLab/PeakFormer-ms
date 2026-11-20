# PeakFormer

PeakFormer identifies molecular formulas from tandem mass spectra by ranking
formula candidates. It builds a feature pyramid in peak-order space, fuses
global stoichiometric context through a top-down pathway, and uses a
millidalton-scale relative-position bias for isobaric discrimination. A
cross-attention ranker scores an entire candidate pool from shared encoder
representations.

This repository contains the PeakFormer model implementation:

```
peakformer/
  model_v7.py    # PeakFormerV7 encoder, CandidateScorer, configuration
  model_v3.py    # pyramid, attention, FiLM and normalization modules
  features.py    # peak features and relative mass edges
```

## Dataset

The KB_MS-Formula benchmark is archived on Zenodo:

- DOI: https://doi.org/10.5281/zenodo.24385348
- 201,012 centroided MS/MS spectra, 20,236 molecules
- 26,543 negative-ion spectra
- leakage-proof train/validation/test splits and a formula-disjoint test subset

See `DATASET.md` for a short pointer and the Zenodo record for the full data
card, metadata, splits, license, and checksums.

## Citation

If you use PeakFormer or KB_MS-Formula, please cite the accompanying
manuscript and the dataset record:

KB_MS-Formula: a dual-polarity tandem mass spectra dataset for molecular
formula identification (Version 1.0). Zenodo.
https://doi.org/10.5281/zenodo.24385348

## License

MIT License. See `LICENSE`.

## Questions

Please open an issue on GitHub:
https://github.com/YjeLab/PeakFormer-ms/issues