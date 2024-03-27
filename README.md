


## Introduction
Open source datasets for paper [Enhance Reasoning for Large Language Models in the Game Werewolf](https://arxiv.org/pdf/2402.02330.pdf)

## Datasets
Links:
[download datasets](https://drive.google.com/file/d/1pw6uIPdjfxssEPELA-U6neejmZ2sIrpe/view?usp=sharing)


## Usage
1. Unzip and move files
```shell
unzip werewolf_data.zip
mv werewolf_data ./data
```

2. Simulation games with human data
```shell
python3 processor/check_data_en.py --path_processed data/werewolf_data
```

3. Simulation games with demo data
```shell
cd processor
python3 check_data_en.py
```

## Output
Werewolf game simulation video

![demo](demo/demo_gif.gif)