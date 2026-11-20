# KB_MS-Formula dataset

The dataset is archived on Zenodo:

https://doi.org/10.5281/zenodo.24385348

The Zenodo record contains:

- `spectra.mgf.gz.part1` and `spectra.mgf.gz.part2` (reassemble, then gunzip)
- `metadata.csv` with one row per spectrum
- `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
- the data card, license, source-license notes, and SHA-256 checksums

At a glance: 201,012 centroided MS/MS spectra, 20,236 molecules, 26,543
negative-ion spectra, leakage-proof splits, and a formula-disjoint test subset
of 16,730 spectra.

The derived benchmark is released under CC BY 4.0. Source spectra retain the
terms of their original libraries; see the Zenodo data card and
`SOURCE_LICENSES.md` in the record.