# python run_classifier.py \
#   --task_name isa \
#   --cache_dir /run/user/1051/ \
#   --do_train \
#   --do_eval \
#   --do_lower_case \
#   --data_dir ./data/project2_data/ \
#   --bert_model bert-base-uncased \
#   --max_seq_length 64 \
#   --train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --num_train_epochs 3.0 \
#   --output_dir modelA/

python run_classifier.py \
  --task_name isa \
  --cache_dir /run/user/1051/ \
  --do_eval \
  --do_lower_case \
  --data_dir ./data/project2_data/ \
  --bert_model modelA \
  --max_seq_length 64 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir modelA/
