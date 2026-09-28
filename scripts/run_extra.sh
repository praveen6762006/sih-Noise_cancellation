#!/usr/bin/env bash
# Extra run: NLMS before the network, trained with the NLMS in the loop. Runs after the main chain.
cd "$(dirname "$0")/.."
python scripts/train.py --name E_nlmspre_s0 --steps ${STEPS:-6000} --batch 16 --workers 10 --val-every 1000 \
    --size small --pretrained --nlms-pre --seed 0 > results/train_E_nlmspre_s0.log 2>&1
tail -2 results/train_E_nlmspre_s0.log | grep -E "done|Error|Traceback"
echo EXTRA_DONE
