# Pre-submission archival release checklist

Complete these steps only after confirming that the manuscript PDF and the
repository artifacts report the same results.

1. Confirm that `requirements.txt`, `data/development_rows.npy`,
   `results/router_vft_gpr_audit/`, `docs/REPRODUCIBILITY.md`, and the script
   used to produce the audit are included in the release commit.
2. Run the documented audit from clean public-source downloads and compare its
   DOI-macro errors, paired router--VFT statistics, and timing records with the
   manuscript tables and figures.
3. Choose and add a code licence. Do not state that the repository is open
   source until a licence has been selected.
4. The `v1.0.1` GitHub Release is published at
   `https://github.com/M1ller-cn/Data-Efficient_CCE/releases/tag/v1.0.1`.
   Do not modify its release artifacts after Zenodo archives the version.
5. Zenodo archived the release as v1.0.1 with the version-specific DOI
   `10.5281/zenodo.22266259`:
   `https://doi.org/10.5281/zenodo.22266259`.
6. Replace the manuscript's Data and Code Availability statement with that
   version-specific DOI, rebuild the PDF, and upload the matching PDF and
   source package.

The third-party source measurement tables are not release artifacts unless
their licences permit redistribution. Their source records and citation terms
remain authoritative.
