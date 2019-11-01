# AMR-parser
Code for our **EMNLP2019** paper, 

**Core Semantic First: A Top-down Approach for AMR Parsing**. [[arxiv\]](https://arxiv.org/pdf/1909.04303.pdf)

Deng Cai and Wai Lam.

# Requirement

python2==Python2.7

python3==Python3.6

`sh setup.sh` (install dependencies)

# Preprocessing

in the directory `preprocessing`
1. make a directory, for example `preprocessing/2017`. Put files `train.txt`, `dev.txt`,`test.txt`in it. The format of these files is shown in our example `preprocessing/data/dev.txt`.
2. `sh go.sh` (you may make necessary changes in `convertingAMR.java`)
3. `python2 preprocess.py`

- We already provided the alignment information (LDC2017T10) as in `common/out_standford`) given by the [Oneplus/tamr](https://github.com/Oneplus/tamr).


# Training
in the directory  `parser`
2. `python3 extract.py && mv *vocab *table ../preprocessing/2017/.` Make vocabularies for the dataset in `../preprocessing/2017/` (you may make necessary changes in `extract.py`)
2. `sh train.sh` Be patient! Checkpoints will be saved in the directory `ckpt` (make necessary changes in `train.sh`).

# Testing

in the directory  `parser`
1. `sh work.sh` (make necessary changes in `work.sh`)
   The most important argument is `--load_path`, set it to the checkpoint file. The output file will be in the same folder with the checkpoint file. For example, if the `load_path` is `somewhere/some_ckpt`, the output file is `somewhere/some_ckpt_test_out`

# Evaluation

in the directory `amr-evaluation-tool-enhanced`
1. `python2 smatch/smatch.py --help`
  A large of portion of the code under this directory is borrowed from [ChunchuanLv/amr-evaluation-tool-enhanced](https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced), we add more options as follows.

  ```shell
    --weighted           whether to use a weighted smatch or not
    --levels LEVELS      how deep you want to evaluate, -1 indicates unlimited, i.e., full graph
    --max_size MAX_SIZE  only consider AMR graphs with limited size <=, -1 indicates unlimited
    --min_size MIN_SIZE  only consider AMR graphs with limited size >=, -1 indicates unlimited
  ```

 For examples:

2. To calculate the smatch-weighted metric in our paper.

   `python2 smatch/smatch.py --pr -f parsed_data golden_data --weighted`

3. To calculate the smatch-core metric in our paper

   `python2 smatch/smatch.py --pr -f parsed_data golden_data --levels 4`