<div align="center">

**Enhancing next token prediction based pre-training for jet foundation models**

Joschka Birk, Anna Hallin, Gregor Kasieczka, Nikol Madzharova, Ian
Pang, David Shih

[![arXiv](https://img.shields.io/badge/arXiv-2512.04149-b31b1b.svg)](https://arxiv.org/abs/2512.04149)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.5-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

> Next token prediction is an attractive pre-training task for jet foundation models, in that it is simulation free and enables excellent generative capabilities that can transfer across datasets. Here we study multiple improvements to next token prediction, building on the initial work of OmniJet-α. Instead of tokenizing particles and subsequently only using the token-ID as the model input for both the generative and the classification task, we adopt a hybrid setup, which allows us to use continuous feature vectors as model input while only using token-IDs in the next token prediction target. Secondly, we explore a combined pre-training strategy that combines masked particle modeling and generative learning objectives. Taken together, these changes greatly improve the performance in downstream classification tasks without any loss in generative performance.

## Dataset

Instructions on how to download the datasets can be found in the repository
[jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer).

## Installation

The recommended (and by us tested) way of running the code is to use the
provided Docker image
[`jobirk/gabbro`](https://hub.docker.com/repository/docker/jobirk/gabbro/general).
The requirements listed in `docker/requirements.txt` are installed in the `conda` environment
`base` of the base image (official pytorch image).
Thus, you have to make sure that the `conda` environment is activated when running the code,
which can be done with `source /opt/conda/bin/activate`.

An interactive session inside a container can be started by running the following command:

```shell
# on a machine with Singularity
singularity shell docker://jobirk/gabbro:latest  # start a shell in the container
source /opt/conda/bin/activate  # activate the conda environment in the container
#
# on a machine with Docker
docker run -it --rm jobirk/gabbro:latest bash  # start a shell in the container
source /opt/conda/bin/activate  # activate the conda environment in the container
```

Alternatively, you can install the requirements from the `docker/requirements.txt` file, but
you'll have to add `pytorch` to the list of requirements, since this is not
included in the `requirements.txt` file (we use the official pytorch image as
base image).

Furthermore, you'll have to add/create a `.env` file in the root of the project
with the following content:

```bash
# stuff for hydra
LOG_DIR="<path to log dir>"
COMET_API_TOKEN="<your comet api token>"
HYDRA_FULL_ERROR=1
```

## Tokenization

You can run the training of the VQ-VAE model by running the following command:

```bash
python gabbro/train.py experiment=example_experiment_tokenization_transformer
```

To create the tokenized dataset, you can run the following command (with adjusted paths):

```shell
python scripts/create_tokenized_jetclass_files.py \
  --ckpt_path=/path/to/vqvae/checkpoint \
  --n_files_train=<number_of_training_files_per_jet_type> \
  --n_files_val=<number_of_validation_files_per_jet_type> \
  --n_files_test=<number_of_test_files_per_jet_type> \
  --output_suffix=<suffix_if_wanted> \
  --jet_types=ZJetsToNuNu_,TTBar_ \
  --output_base_dir=/data/dust/user/birkjosc/datasets/jetclass_tokenized \
  --input_base_dir=/data/dust/user/birkjosc/datasets/jetclass/JetClass \
  --dataset_type=jetclass \
  --shuffle_particles
```

Afterwards, the tokenized dataset will be saved in a subdirectory of the
`output_base_dir` directory.

## Joint Pre-Training

If you want to run a training of the backbone with the dedicated head(s), you
first have to create the tokenized dataset (see above).
Note that you have to make sure that the checkpoint of the tokenizer is saved/copied
to that directory as `model_ckpt.ckpt` and the training config as `config.yaml`
(this is necessary since the gen. training will look for those files to reconstruct
tokens back to physical space).

You can then run the joint Generative+MPM training of the model by running the
following command:

```shell
python gabbro/train.py \
    'experiment=[ example_experiment_backbone_and_head ]' \
    'model.loss_term_weights={mpm:1, gen:1, class:0}'
```

## Classification Training

To fine-tune a pre-trained backbone on the jet classification task, you can run the
following command:

```shell
python gabbro/train.py \
    'experiment=[ example_experiment_backbone_and_head ]' \
    'model.loss_term_weights={mpm:0, gen:0, class:1}' \
    'model.causal_bidirectional_hybrid=false' \
    'model.backbone_cfg.apply_causal_mask=false'
```
