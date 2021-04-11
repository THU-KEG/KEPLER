MAX_UPDATES=7000      # Number of training steps.
WARMUP_UPDATES=700    # Linearly increase LR over this many steps.
LR=3e-05            # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=16      # Batch size.
SEED=2333             # Random seed.
ROBERTA_PATH=path_to_KEPLER_original_checkpoint     #Path to the original KEPLER checkpoint


UPDATE_FREQ=1
DATA_DIR=path_to_data        #Path to preprocessed TACRED data
```
The data path should contain five files

The train.jsonl, valid.jsonl and test.jsonl are splited TACRED data, each line is a json string like the following example:
{
    "token": ["Zagat", "Survey", ",", "the", "guide", "empire", "that", "started", "..."], #tokenized sentence
    "h": {"name": "Zagat", "pos": [0, 1]}, #head entity
    "t": {"name": "1979", "pos": [17, 18]}, #tail entity
    "relation": "org:founded" # relation name
}

The rel2id.json is a dict mapping relation names to numberical ids

The dict.txt is the RoBERTa dictionary file
```
CHECKPOINT_PATH=path_to_save_checkpoint     #Path to save the checkpoints finetuned for TACRED

fairseq-train --fp16 \
    $DATA_DIR \
    --restore-file $ROBERTA_PATH \
    --save-dir $CHECKPOINT_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
    --best-checkpoint-metric f1 --maximize-best-checkpoint-metric \
    --task tacred --init-token 0 --bpe gpt2 \
    --arch roberta_base --max-positions 512 \
    --gpt2-encoder-json gpt2_bpe/encoder.json --gpt2-vocab-bpe gpt2_bpe/vocab.bpe \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion relation_extraction --num-classes 42 --negative-label 13\
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES --total-num-update $MAX_UPDATES \
    --max-sentences $MAX_SENTENCES \
    --max-update $MAX_UPDATES \
    --update-freq $UPDATE_FREQ \
    --log-format simple --log-interval 25 \
    --seed $SEED \
    --num-workers 0 --ddp-backend=no_c10d \
    --find-unused-parameters \
    --save-interval-updates 200