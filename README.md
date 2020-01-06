# Deeper Insights into Weight Sharing in Neural Architecture Search

Experiments Code. Modified from DARTS.

## Instructions

For NAS experiments, run

```
python search.py --name xxx --config_file /path/to/config
```

For finetuning on checkpoints, run this after you run NAS (possibly after cutoff)

```
python finetune.py --name xxx --config_file /path/to/config
```

For more instructions on how to use search/finetune, run

```
python search.py -h
```

## Run on Local (with multiple GPUs)

This will launch 10 jobs with seed from 0 to 9.

```
python submit_to_v100.py --config_file /path/to/config --tune_key seed --tune_range 10
```

## Run on OpenPAI cluster

```
python submit_to_pai.py /path/to/config --nodes 4 --repeat 10
```
