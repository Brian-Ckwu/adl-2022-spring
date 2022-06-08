for target in relevance aggressiveness overall
do
  python train_qual_regressor.py \
    --target_score $target \
    --model_type xlnet \
    --model_name xlnet-base-cased \
    --lowercase false \
    --device 1
done