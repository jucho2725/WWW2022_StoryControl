
## NOTE: We have not uploaded the dataset or model checkpoint yet. We will upload those and update the instructions soon!

Code for paper - [Genre-Controllable Story Generation via Supervised Contrastive Learning]()

### Install and Data preprocess
The code is implemented on [transformers v4.3.1](), follow the same steps to install and prepare the processed dataset.


> Step 1: Install prerequisites.

```bash
## you may want to build a virtual environment first.
git clone https://github.com/jucho2725/WWW2022_StoryControl.git
cd leca
pip install -r requirements.txt
```

> Step 2: Process dataset

I will updates the dataset link 

### Run experiment

> Step 1: Train genre classifier.

See scripts/run_cls.sh. You may need to revise the variables in the shell scripts first according to your case. 

> Step 2: Train genre-controllable generator and evaluate the perfomance by the classifier.

See scripts/run_gen_cls.sh. You may need to revise the variables in the shell scripts first according to your case.

### Citation

TBD

<!-- ```bibtex
@inproceedings{chen2020leca,
  title     = {Lexical-Constraint-Aware Neural Machine Translation via Data Augmentation},
  author    = {Chen, Guanhua and Chen, Yun and Wang, Yong and Li, Victor O.K.},
  booktitle = {Proceedings of {IJCAI} 2020: Main track},          
  pages     = {3587--3593},
  year      = {2020},
  month     = {7},
}
``` -->