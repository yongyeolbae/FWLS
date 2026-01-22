Goal

Compare performance changes by switching only the loss under the same data split and model setting.

Pure: BCE, Dice

Weighted: WBCE, WDice (pixel-wise weight map)

Structure

configs/ : experiment configs (yaml) — paths, hyperparams, output dir, etc.

scripts/ : training/evaluation code

outputs/ : saved results (keep empty in git; ignore recommended)

Where the “core code” lives

Model usage

model = ... / logits = model(img) in each script

PSCCNet-Lite: model(img) → [m1,m2,m3,m4] (deep supervision logits list)

Weighted losses (WBCE/WDice)

weight-map build: build_importance_weight_map_*

DS resize for weights: build_ds_weights(...)

WBCE/WDice implementations: WeightedBCEWithLogitsLoss, WeightedDiceLoss

selected in training loop: train_one_epoch(..., loss_mode="bce|wbce|dice|wdice")


Run
bash run.sh
or
python scripts/train_psccnet_lite.py --config configs/psccnet_pair_rule.yaml

