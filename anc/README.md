# Reproducibility package

This directory accompanies *Privacy Failure in Split-LLM Training: The
Returned Gradient Nullifies the Decoys*. It contains the executable
implementation, frozen protocols, compact
derived result records, and figure-generation code used by the paper.

The package has two deliberately different reproducibility levels:

1. **Artifact-level reproduction.** The calibration and leak-shape figures can
   be regenerated from the included JSON records. Reported per-seed and paired
   values can be inspected or recomputed from the included derived summaries.
2. **Experimental rerun.** The training, capture, attack, and scoring code is
   included, but rerunning the GPU experiments additionally requires the model,
   corpus, TLS credentials, and a two-node GPU environment. Those large or
   secret inputs are not distributed here.

This is a focused release, not a copy of the development repository.

## Directory map

- `attacker/`: frozen attacker framework and attack families.
- `bin/`: experiment drivers, capture conversion, scoring, statistics, and
  validation scripts.
- `privacy_runtime/`: runtime privacy mechanisms.
- `split-training/`: trusted- and remote-side model implementations.
- `paper-data/`: frozen protocols, evidence ledger, compact result JSONs, and
  claim/figure inputs. `paper-data/provenance/e1_source_of_record/` preserves
  the historical E1 runtime source snapshot and its manifest.
- `outputs/`: compact positive-control and isolation-audit summaries.
- `provenance/`: launch scripts retained for the cited campaigns. Large logs
  are omitted.
- `docs/experiments/`: protocols and experiment-specific methodological notes.
- `docs/audits/`: traceability and red-team audit records cited by the paper.
- `papers/paper-1/figs/build_figures.py`: regenerates the two external PDF
  figures from `paper-data/`.
- `env/requirements-container.txt`: package snapshot from the environment of
  record. It is a drift record, not a portable lockfile.

## Quick checks

Run these commands from the `anc/` directory.

Validate the frozen evaluation protocol:

```bash
python3 bin/validate_evaluation_protocol.py \
  --protocol paper-data/evaluation_protocol.json
```

Regenerate the two plotted figures from the committed summaries:

```bash
cd papers/paper-1/figs
python3 build_figures.py
```

This second command requires Python and Matplotlib. It writes
`fig_w24_dose.pdf` and `fig_w56_shape.pdf` beside the script and prints their
SHA-256 digests.

The seed-44 headline trace is:

```text
paper-data/collected/diagnostic/e1_reproduction_w12/
  e1_repro_w12_s44_arm_grad_real_paired.json
```

Read `best_eligible.paired_advantage_pp`; the recorded value is `0.6929` pp.
The adjacent `...grad_real_shuffled_paired.json` is the matched negative
control. `bin/paired_advantage.py` implements the frame-clustered paired
statistic used for those files.

## Re-running a headline experiment

The primary unprotected-backward-channel driver is:

```text
bin/e1_unprotected_cell.sh
```

The utility-passing defended/undefended matrix driver is:

```text
bin/phasec_defended_cell.sh
```

Both call `bin/run_latent_native_v5_06b.py`, the remote server under
`split-training/`, the `attacker` package, and the paired scorer. The scripts
were run in an NVIDIA PyTorch 26.02-derived container with Python 3.12.3,
PyTorch 2.11.0a0+eb65b36914.nv26.02, CUDA 13.1, and Transformers 5.13.0.
The full observed package set is in `env/requirements-container.txt`.

The scripts expect a container mount rooted at `/workspace/experiments` with:

```text
models/qwen3-0.6b/
models/wikitext2_corpus.txt
tls/ca.crt
results/training/
```

Model and corpus identifiers/hashes from the run of record are recorded in
`paper-data/corpus_manifest_original.json` and the per-cell metadata. The model
weights and corpus are not redistributed; obtain them under their own licenses
and verify the recorded hashes. TLS private keys are intentionally excluded.
Set `CLOUD_URL` to the remote server endpoint. The launch scripts under
`provenance/` retain the original cluster path conventions and should be
treated as provenance templates, not executed unchanged on a new system.

The environment snapshot cannot rebuild the exact 21.2 GB container by itself:
the original image included NVIDIA-patched framework builds and no complete
Dockerfile was frozen. A faithful rerun therefore requires constructing an
equivalent NVIDIA GPU environment and recording any drift.

## Included evidence

The compact release includes:

- six packaged reproduction seeds and three post-freeze confirmation seeds;
- the twelve-seed hierarchical summary;
- the matched-pair record and frozen confirmation-matrix specification;
- representation-matched positive-control summaries;
- the W2.4 dose-response calibration records;
- the W5.6 nine-cell leak-shape records;
- Phase-C per-cell JSON summaries and the aggregate paired summary;
- complete-transcript manifest, coverage summary, and verified-collection
  record; and
- audit records identifying which values are directly re-derivable and which
  are supported only by committed derived summaries.

## Deliberate omissions and limits

The following files are not in this arXiv-sized package:

- raw `.pt` capture, bundle, checkpoint, and prediction tensors;
- row-level `.jsonl` prediction logs;
- the 7.4 GB Phase-C bundle store;
- the 88 MB `w34_index.json` and the W3.4 checkpoint archive;
- the host-only 180,636-event raw transcript;
- TLS private keys and certificates; and
- the independent internal scorer implementation and its prediction tensors.

Consequently, the audit-cell and dose-response plots are reproducible from this
package, while several headline and mitigation values are **inspectable and
redisplayable from committed derived JSONs but not end-to-end re-derivable from
raw tensors here**. This matches the availability boundary stated in the
paper's artifact appendix. A full archival release should place the omitted
non-secret raw artifacts in a versioned data repository and cite its DOI.

## Integrity

`SHA256SUMS.txt` covers every file in this ancillary directory except the hash
list itself. Verify it from `anc/` with:

```bash
shasum -a 256 -c SHA256SUMS.txt
```
