# Golden gates (gates/run.py). Pass the venv explicitly in a checkout without .venv:
#   make check PYTHON=<path to venv>/bin/python
# The freeze-and-retrain gates use a synthetic source checkpoint by default; with the two
# trained source runs under $DUALSPS_OUT_ROOT/out/, `make check FRZ=real` checks those too.
PYTHON ?= $(firstword $(wildcard .venv/bin/python) python)
FRZ ?= synthetic
GPU ?= 0

.PHONY: check g1 g2 g3 g4 g5 g7 smoke paper snapshot
check:
	$(PYTHON) gates/run.py all --frz-source=$(FRZ)

g1 g2 g3 g4 g5 g7:
	$(PYTHON) gates/run.py $@ --frz-source=$(FRZ)

# A few optimizer steps of scripts/train.py for every model family, only to show that
# training runs: on CPU at reduced width (about a minute), or with GPU=1 on one GPU at the
# paper's model size.
smoke:
	$(PYTHON) scripts/smoke_train.py $(if $(filter 1,$(GPU)),--gpu)

# The paper's tables, text numbers and figures from the snapshotted results (CPU, about a
# minute): paper/tables/ (with numbers.tex) and paper/figures/. After a model config or FLOP-accounting change run
#   $(PYTHON) scripts/analysis/paper_data.py --refresh-arch-stats
# first. See scripts/analysis/README.md.
paper:
	$(PYTHON) scripts/analysis/make_tables.py
	$(PYTHON) scripts/analysis/paper_figures.py

# Copy the live ledger and wall-clock rows ($DUALSPS_OUT_ROOT/results/{ledger,wallclock}.jsonl,
# where eval_runs.py and bench_wallclock.py append) over the snapshot the paper is built from.
snapshot:
	$(PYTHON) scripts/dualsps/ledger.py --snapshot
