#!/usr/bin/env bash
CMD="python train_semi-supervised.py"
JOB_NAME="tr_c3ps_same_split_with_magic"
submit_slurm \
  --command "${CMD}" \
  -t 24:00:00 \
  -p medium \
  -c 16 \
  -m 54G \
  -n "${JOB_NAME}" \
  -e ssl_seg \
  --constraint "vr40g|vr80g|vr144g"
