export PREDICT_FILE=$1

export BERT_BASE_DIR=multi_cased_L-12_H-768_A-12
export OUTPUT_DIR=$2

echo "Predict file: $PREDICT_FILE"
echo "BERT base dir: $BERT_BASE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo

python3 bert/run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --do_predict=True \
  --predict_file=$PREDICT_FILE \
  --do_lower_case=False \
  --num_train_epochs=2.0 \
  --max_seq_length=320 \
  --save_checkpoints_steps=8000 \
  --train_batch_size=10 \
  --output_dir=$OUTPUT_DIR
