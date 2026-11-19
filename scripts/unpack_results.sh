#!/bin/bash
# The model outputs are stored gzipped (*.jsonl.gz) to keep the repository small. Unpack them before running any
# analysis script; the .gz files are kept. Run from the repository root.
find results numword/results swthink/results rank_study/results -name '*.jsonl.gz' -exec gunzip -kf {} +
echo "unpacked $(find results numword/results swthink/results rank_study/results -name '*.jsonl' | wc -l) files"
