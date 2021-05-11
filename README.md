# KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation

Source code for TACL 2021 paper [KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00360/98089/KEPLER-A-Unified-Model-for-Knowledge-Embedding-and).

## Preliminaries

- This code is developed on top of [fairseq](https://github.com/pytorch/fairseq). Refer to [its documentation](/fairseqREADME.md) for the installation and basic usage.
- Pre-training requires the [Wikidata5M dataset](https://deepgraphlearning.github.io/project/wikidata5m).

## Pre-training

### Preprocessing

- Refer to [this document](examples/roberta/README.pretraining.md) for the detailed data preprocessing of the datasets used in the Masked Language Modeling (MLM) objective.
- For the data preprocessing of datasets in the Knowledge Embedding (KE) objective. We (1) do negative sampling for the training triplets and dump the corresponding entity descriptions into separated files, each line is for one (negative) triplet; (2) split the files to align with the number of training instances of the MLM dataset and so that get data files for each epoch; (3) then BPE tokenize and binarize the files similarly to the MLM datasets. We will soon clean this process into a unified script.

### Running

An example pre-training script:

```bash
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
LR=6e-04                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=3        # Batch size.
ROBERTA_PATH="path/to/roberta.base/model.pt" #Path to the original roberta model
CHECKPOINT_PATH="path/to/checkpoints" #Directory to store the checkpoints
UPDATE_FREQ=`expr 784 / $SLURM_JOB_NUM_NODES` # Increase the batch size

DATA_DIR=../Data

#Path to the preprocessed KE dataset, each item corresponds to a data directory for one epoch
KE_DATA=$DATA_DIR/KEI/KEI1_0:$DATA_DIR/KEI/KEI1_1:$DATA_DIR/KEI/KEI1_2:$DATA_DIR/KEI/KEI1_3:$DATA_DIR/KEI/KEI3_0:$DATA_DIR/KEI/KEI3_1:$DATA_DIR/KEI/KEI3_2:$DATA_DIR/KEI/KEI3_3:$DATA_DIR/KEI/KEI5_0:$DATA_DIR/KEI/KEI5_1:$DATA_DIR/KEI/KEI5_2:$DATA_DIR/KEI/KEI5_3:$DATA_DIR/KEI/KEI7_0:$DATA_DIR/KEI/KEI7_1:$DATA_DIR/KEI/KEI7_2:$DATA_DIR/KEI/KEI7_3:$DATA_DIR/KEI/KEI9_0:$DATA_DIR/KEI/KEI9_1:$DATA_DIR/KEI/KEI9_2:$DATA_DIR/KEI/KEI9_3:

DIST_SIZE=`expr $SLURM_JOB_NUM_NODES \* 4`

fairseq-train $DATA_DIR/MLM \                #Path to the preprocessed MLM datasets
        --KEdata $KE_DATA \                      #Path to the preprocessed KE datasets
        --restore-file $ROBERTA_PATH \
        --save-dir $CHECKPOINT_PATH \
        --max-sentences $MAX_SENTENCES \
        --tokens-per-sample 512 \
        --task MLMetKE \                     
        --sample-break-mode complete \
        --required-batch-size-multiple 1 \
        --arch roberta_base \
        --criterion MLMetKE \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
        --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_UPDATES --warmup-updates $WARMUP_UPDATES \
        --update-freq $UPDATE_FREQ \
        --negative-sample-size 1 \ # Negative sampling size (one negative head and one negative tail)
        --ke-model TransE \ 
        --init-token 0 \
        --separator-token 2 \
        --gamma 4 \        # Margin of the KE objective
        --nrelation 822 \
        --skip-invalid-size-inputs-valid-test \
        --fp16 --fp16-init-scale 2 --threshold-loss-scale 1 --fp16-scale-window 128 \
        --reset-optimizer --distributed-world-size ${DIST_SIZE} --ddp-backend no_c10d --distributed-port 23456 \
        --log-format simple --log-interval 1 \
        #--relation-desc  #Add this option to encode the relation descriptions as relation embeddings (KEPLER-Rel in the paper)
```

**Note:** The above command assumes distributed training on 64x16GB V100 GPUs, 16 machines. If you have fewer GPUs or GPUs with less memory you may need to reduce `$MAX_SENTENCES` and increase `$UPDATE_FREQ` to compensate. Alternatively if you have more GPUs you can decrease `$UPDATE_FREQ` accordingly to increase training speed.

**Note:** If you are interested in the detailed implementations. The main implementations are in [tasks/MLMetKE.py](fairseq/tasks/MLMetKE.py) and [criterions/MLMetKE.py](fairseq/criterions/MLMetKE.py). We encourage to master the fairseq toolkit before learning KEPLER implementation details.

## Usage for NLP Tasks

We release the pre-trained [checkpoint for NLP tasks](https://cloud.tsinghua.edu.cn/f/e03f7a904526498c81a4/?dl=1). Since KEPLER does not modify RoBERTa model architectures, the KEPLER checkpoint can be directly used in the same way as RoBERTa checkpoints in the downstream NLP tasks.

### Convert Checkpoint to HuggingFace's Transformers

In the fine-tuning and usage, it will be more convinent to convert the original fairseq checkpoints to [HuggingFace's Transfomers](https://github.com/huggingface/transformers).

The conversion can be finished with [this code](https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/convert_roberta_original_pytorch_checkpoint_to_pytorch.py). The example command is:

```bash
python -m transformers.convert_roberta_original_pytorch_checkpoint_to_pytorch.py \
			--roberta_checkpoint_path path_to_KEPLER_checkpoint \
			--pytorch_dump_folder_path path_to_output \
```

### TACRED

We suggest to use the converted HuggingFace's Transformers checkpoint as well as the [OpenNRE](https://github.com/thunlp/OpenNRE) library to perform experiments on TACRED. An example code will be updated soon.

To directly fine-tune KEPLER on TACRED in fairseq framework, please refer to [this script](examples/KEPLER/TACRED/TACRED.sh). The script requires 2x16GB V100 GPUs.

### FewRel

To finetune KEPLER on FewRel, you can use the offiicial code in the [FewRel repo](https://github.com/thunlp/FewRel) and set `--encoder roberta` as well as `--pretrained_checkpoint path_to_converted_KEPLER`.

### OpenEntity

Please refer to [this directory](examples/KEPLER/OpenEntity) and [this script](examples/KEPLER/OpenEntity/run_openentity.sh) for the codes of OpenEntity experiments.

These codes are modified on top of [ERNIE](https://github.com/thunlp/ERNIE).

### GLUE

For the fine-tuning on GLUE tasks, refer to the [official guide of RoBERTa](examples/roberta/README.glue.md).

Refer to [this directory](examples/KEPLER/GLUE) for the example scripts along with hyper-parameters.

### Knowledge Probing (LAMA and LAMA-UHN)

For the experiments on LAMA, please refer to the codes in the [LAMA repo](https://github.com/facebookresearch/LAMA) and set `--roberta_model_dir path_to_converted_KEPLER`.

The LAMA-UHN dataset can be created with [this scirpt](https://github.com/facebookresearch/LAMA/blob/master/scripts/create_lama_uhn.py).

## Usage for Knowledge Embedding

We release the pre-trained [checkpoint for KE tasks](https://cloud.tsinghua.edu.cn/f/749183d2541c43a08568/?dl=1).

First, install the `graphvite` package in `./graphvite` following its instructions. `graphvite` is a fast evaluation tool for network embedding and knowledge embedding, and we made some modification to suit our need.

Then, download the Wikidata5m dataset from [this website](https://deepgraphlearning.github.io/project/wikidata5m).

The next step is to use `ke_tool/evaluate_transe_transductive.py` and `ke_tool/evaluate_transe_inductive.py` for KE evaluation. The arguments are as following,

* `--entity_embeddings`: a numpy file storing the entity embeddings.
* `--relation_embeddings`: a numpy file storing the relation embeddings.
* `--dim`: the dimension of the relation and entity embeddings.
* `--entity2id`: a json file that maps entity names (in the dataset) to the ids in the entity embedding numpy file, where the key is the entity names in the dataset, and the value is the id in the numpy file.
* `--relation2id`: a json file that maps relation names (in the dataset) to the ids in the relation embedding numpy file.
* `--dataset`: the test data file.
* `--train_dataset`: the training data file (only for transductive setting).
* `--val_dataset`: the validation data file (only for transductive setting).


## Citation

If the codes help you, please cite our paper:

**KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation.** *Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan Zhang, Zhiyuan Liu, Juanzi Li, Jian Tang.* TACL 2021.
