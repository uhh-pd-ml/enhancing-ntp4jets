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