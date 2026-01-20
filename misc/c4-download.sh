#!/bin/bash
set -x
trap 'exit 1' INT

for i in {00000..01023}; do
    wget -c -O "c4-train.$i-of-01024.json.gz" "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.$i-of-01024.json.gz?download=true"
done
