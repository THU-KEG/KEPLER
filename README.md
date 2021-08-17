# KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation

Source code for TACL 2021 paper [KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00360/98089/KEPLER-A-Unified-Model-for-Knowledge-Embedding-and).

## Requirements

- [PyTorch](http://pytorch.org/) version >= 1.1.0
- Python version >= 3.5
- For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
- **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` option

 ## Installation

This repo is developed on top of [fairseq](https://github.com/pytorch/fairseq) and you can install our version like installing fairseq from source:

```bash
pip install cython
git clone https://github.com/THU-KEG/KEPLER
cd KEPLER
pip install --editable .
```

## Pre-training

### Preprocessing for MLM data

Refer to [the RoBERTa document](examples/roberta/README.pretraining.md) for the detailed data preprocessing of the datasets used in the Masked Language Modeling (MLM) objective.

### Preprocessing for KE data

<span id="KEpre">The pre-training with KE objective requires the [Wikidata5M dataset](https://deepgraphlearning.github.io/project/wikidata5m). Here we use the transductive split of Wikidata5M to demonstrate how to preprocess the KE data. The scripts used below are in [this folder](examples/KEPLER/Pretrain/). </span>

Download the Wikidata5M transductive data and its corresponding corpus, and then uncompress them:

```bash
wget -O wikidata5m_transductive.tar.gz https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1
wget -O wikidata5m_text.txt.gz https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1
tar -xzvf wikidata5m_transductive.tar.gz
gzip -d wikidata5m_text.txt.gz
```

Convert the original Wikidata5M files into the numerical format used in pre-training:

```bash
python convert.py --text wikidata5m_text.txt \
		--train wikidata5m_transductive_train.txt \
		--valid wikidata5m_transductive_valid.txt \
		--converted_text Qdesc.txt \
		--converted_train train.txt \
		--converted_valid valid.txt
```

Encode the entity descriptions with the GPT-2 BPE:

```bash
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
python -m examples.roberta.multiprocessing_bpe_encoder \
		--encoder-json gpt2_bpe/encoder.json \
		--vocab-bpe gpt2_bpe/vocab.bpe \
		--inputs Qdesc.txt \
		--outputs Qdesc.bpe \
		--keep-empty \
		--workers 60
```

Do negative sampling and dump the whole training and validation data:

```bash
python KGpreprocess.py --dumpPath KE1 \
		-ns 1 \
		--ent_desc Qdesc.bpe \
		--train train.txt \
		--valid valid.txt
```

The above command generates training and validation data for **one** epoch. You can generate data for more epochs by running it many times and dump to different folders (e.g. `KE2`, `KE3`, ...). 

There may be too many instances in the KE training data generated above and thus results in the time for training one epoch is too long. We then randomly split the KE training data into smaller parts and the number of training instances in each part aligns with the MLM training data:

```bash
python splitDump.py --Path KE1 \
		--split_size 6834352 \
		--negative_sampling_size 1
```

The `KE1` will be splited into `KE1_0`, `KE1_1`, `KE1_2`, `KE1_3`. We then binarize them for training:

```bash
wget -O gpt2_bpe/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
for KE_Data in ./KE1_0/ ./KE1_1/ ./KE1_2/ ./KE1_3/ ; do \
    for SPLIT in head tail negHead negTail; do \
        fairseq-preprocess \ #if fairseq-preprocess cannot be founded, use "python -m fairseq_cli.preprocess" instead
            --only-source \
            --srcdict gpt2_bpe/dict.txt \
            --trainpref ${KE_Data}${SPLIT}/train.bpe \
            --validpref ${KE_Data}${SPLIT}/valid.bpe \
            --destdir ${KE_Data}${SPLIT} \
            --workers 60; \
    done \
done
```

### Running

An example pre-training script:

```bash
TOTAL_UPDATES=125000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
LR=6e-04                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=3        # Batch size.
NUM_NODES=16					 # Number of machines
ROBERTA_PATH="path/to/roberta.base/model.pt" #Path to the original roberta model
CHECKPOINT_PATH="path/to/checkpoints" #Directory to store the checkpoints
UPDATE_FREQ=`expr 784 / $NUM_NODES` # Increase the batch size

DATA_DIR=../Data

#Path to the preprocessed KE dataset, each item corresponds to a data directory for one epoch
KE_DATA=$DATA_DIR/KEI/KEI1_0:$DATA_DIR/KEI/KEI1_1:$DATA_DIR/KEI/KEI1_2:$DATA_DIR/KEI/KEI1_3:$DATA_DIR/KEI/KEI3_0:$DATA_DIR/KEI/KEI3_1:$DATA_DIR/KEI/KEI3_2:$DATA_DIR/KEI/KEI3_3:$DATA_DIR/KEI/KEI5_0:$DATA_DIR/KEI/KEI5_1:$DATA_DIR/KEI/KEI5_2:$DATA_DIR/KEI/KEI5_3:$DATA_DIR/KEI/KEI7_0:$DATA_DIR/KEI/KEI7_1:$DATA_DIR/KEI/KEI7_2:$DATA_DIR/KEI/KEI7_3:$DATA_DIR/KEI/KEI9_0:$DATA_DIR/KEI/KEI9_1:$DATA_DIR/KEI/KEI9_2:$DATA_DIR/KEI/KEI9_3:

DIST_SIZE=`expr $NUM_NODES \* 4`

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

In the fine-tuning and usage, it will be more convinent to convert the original fairseq checkpoints to [HuggingFace's Transformers](https://github.com/huggingface/transformers).

The conversion can be finished with [this code](https://github.com/huggingface/transformers/blob/master/src/transformers/models/roberta/convert_roberta_original_pytorch_checkpoint_to_pytorch.py). The example command is:

```bash
python -m transformers.convert_roberta_original_pytorch_checkpoint_to_pytorch \
			--roberta_checkpoint_path path_to_KEPLER_checkpoint \
			--pytorch_dump_folder_path path_to_output \
```

The `path_to_KEPLER_checkpoint` should contain `model.pt` (the downloaded KEPLER checkpoint) and `dict.txt` (standard RoBERTa dictionary file).

Note that the new versions of HuggingFace's Transformers library requires `fairseq>=0.9.0`, but the modified fairseq library in this repo and our checkpoints generated with is `fairseq==0.8.0`. The two versions are minorly different in the checkpoint format. Hence `transformers<=2.2.2` or `pytorch_transformers` are needed for checkpoint conversion here.

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

First, install the `graphvite` package in[`./graphvite`](/graphvite) following its instructions. [GraphVite](https://github.com/DeepGraphLearning/graphvite) is an fast toolkit for network embedding and knowledge embedding, and we made some modifications on top of them.

Generate the entity embeddings and relation embeddings with[`generate_embeddings.py`](examples/KEPLER/KE/generate_embeddings.py). The arguments are as following:

* `--data`: the entity decription data, a single file, each line is an entity description. It should be BPE encoded and binarized like introduced in the [**Preprocessing for KE data**](#KEpre)
* `--ckpt_dir`: path of the KEPLER checkpoint.
* `--ckpt`: filename of the KEPLER checkpoint.
* `--dict`: path to the[`dict.txt`](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt) file.
* `--ent_emb`: filename to dump entity embeddings (in numpy format).
* `--rel_emb`: filename to dump relation embeddings (in numpy format).
* `--batch_size`: batch size used in inference.

Then use [`evaluate_transe_transductive.py`](examples/KEPLER/KE/evaluate_transe_transductive.py) and [ `ke_tool/evaluate_transe_inductive.py`](examples/KEPLER/KE/evaluate_transe_inductive.py) for KE evaluation. The arguments are as following:

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

```bibtex
@article{wang2021KEPLER,
  title={KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation},
  author={Xiaozhi Wang and Tianyu Gao and Zhaocheng Zhu and Zhengyan Zhang and Zhiyuan Liu and Juanzi Li and Jian Tang},
  journal={Transactions of the Association for Computational Linguistics},
  year={2021},
  volume={9},
  doi = {10.1162/tacl_a_00360},
  pages={176-194}
}
```

These codes are developed on top of [fairseq](https://github.com/pytorch/fairseq) and [GraphVite](https://github.com/DeepGraphLearning/graphvite):

```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```

```bibtex
@inproceedings{zhu2019graphvite,
    title={GraphVite: A High-Performance CPU-GPU Hybrid System for Node Embedding},
     author={Zhu, Zhaocheng and Xu, Shizhen and Qu, Meng and Tang, Jian},
     booktitle={The World Wide Web Conference},
     pages={2494--2504},
     year={2019},
     organization={ACM}
 }
```
