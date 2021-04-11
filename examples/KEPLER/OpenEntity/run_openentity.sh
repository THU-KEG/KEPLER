export DATA_DIR=path_to_OpenEntity

python run_typing.py \
  --model_type roberta \
  --model_name_or_path path_to_converted_KEPLER \
  --task_name typing \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 40 \
  --output_dir path_to_output_checkpoint \
  --evaluate_during_training \
