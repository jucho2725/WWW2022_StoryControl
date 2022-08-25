
Code for paper - [Genre-Controllable Story Generation via Supervised Contrastive Learning](https://dl.acm.org/doi/10.1145/3485447.3512004)

### Install and Data preprocess
Our experiment environment can be easily reproduced by NVIDIA GPU Container nvcr.io/nvidia/pytorch:21.07-py3. We assume the same docker container setting as beginning.

Follow the same steps to install and prepare the processed dataset.


> Step 1: Install prerequisites.
```bash
## you may want to build a virtual environment first.
## you may need to install pytorch >= 1.7.0 first
git clone https://github.com/jucho2725/WWW2022_StoryControl.git
cd WWW2022_StoryControl
pip install -r requirements.txt
```

> Step 2: Process dataset

[dataset link](https://drive.google.com/file/d/1HPjzTvpKW1WaitGASRR7pE2CadcfO_SD/view?usp=sharing)

Download the dataset and use files as the script described.

### Run experiment

> Step 1: Train genre classifier.

See scripts/1_run_cls.sh. You may need to revise the variables in the shell scripts first according to your case. 

> Step 2: Train genre-controllable generator and evaluate the perfomance by the classifier.

See scripts/2_run_gen.sh. You may need to revise the variables in the shell scripts first according to your case.

### Citation

```
@inproceedings{10.1145/3485447.3512004,
author = {Cho, JinUk and Jeong, MinSu and Bak, JinYeong and Cheong, Yun-Gyung},
title = {Genre-Controllable Story Generation via Supervised Contrastive Learning},
year = {2022},
isbn = {9781450390965},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3485447.3512004},
doi = {10.1145/3485447.3512004},
abstract = {While controllable text generation has received attention due to the recent advances in large-scale pre-trained language models, there is a lack of research that focuses on story-specific controllability. To address this, we present Story Control via Supervised Contrastive learning model (SCSC), to create a story conditioned on genre. For this, we design a supervised contrastive objective combined with log-likelihood objective, to capture the intrinsic differences among the stories in different genres. The results of our automated evaluation and user study demonstrate that the proposed method is effective in genre-controlled story generation.},
booktitle = {Proceedings of the ACM Web Conference 2022},
pages = {2839–2849},
numpages = {11},
keywords = {contrastive learning, controllable text generation, automated story generation, natural language generation},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}
```