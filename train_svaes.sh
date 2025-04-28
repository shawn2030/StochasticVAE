#!/usr/bin/env bash

source /home/rdlvcs/.virtualenvs/StochasticVAE/bin/activate

LAMBDAS=(
  "1.0"
  "1.1"
  "1.3"
  "1.5"
  "2.0"
  "5.0"
  "10.0"
  "100.0"
)

LEARNING_RATES=(
  "1e-5"
  "1e-3"
)

LOGVARS=(
  "-6"
  "-10"
)

DECODER_RUN="37abd9dfafa647ecbdf484d76a04f169"

for LR in "${LEARNING_RATES[@]}"; do
  # First, re-train one VAE with the frozen decoder (so it's maximally comparable with SVAE training)
  echo "Training VAE with frozen decoder"
      python svae/main.py \
        --learning_rate=1e-3 \
        --epochs=500 \
        --lambda="inf" \
        --user_input_logvar="-inf" \
        --learning_rate="$LR" \
        --load_model_from_run="$DECODER_RUN" \
        --init_encoder || exit 1

  for LOGVAR in "${LOGVARS[@]}"; do
    for LAMBDA in "${LAMBDAS[@]}"; do
      echo "Training SVAE with lambda $LAMBDA and logvar $LOGVAR"
      python svae/main.py \
        --lambda="$LAMBDA" \
        --learning_rate=1e-3 \
        --epochs=500 \
        --user_input_logvar="$LOGVAR" \
        --learning_rate="$LR" \
        --load_model_from_run="$DECODER_RUN" \
        --init_encoder || exit 1
    done
  done
done