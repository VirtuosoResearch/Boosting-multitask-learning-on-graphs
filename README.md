# Overview

This repository provided an algorithm as a generic boosting procedure that improves multitask learning on graphs. The algorithm cluster graph learning tasks into multiple groups and train one graph neural network for each group. In the procedure, we first model higher-order task affinities by sampling random task subsets and evaluating multitask performances. Then, we find related task groupings through clustering task affinity scores. 

### Data Preparation

**Community detection.** We provide the datasets for conducting community detection named `data.zip` under the `./data/` folder used. Unzip the file under the folder, then one can directly load them in the code. 

**Molecule property prediction.** We conduct experiments on graph multi-task learning datasets on molecule graph prediction tasks. Our code directly downloads the datasets inside the script. Please pre-install the `ogb` and `torch-geometric` packages. 

<br/>

### **Section 1: Overlapping Community detection**

Use `train_multitask.py` for the experiments of training a GNN on community detection tasks. Please specify the following key parameters:

- `--dataset` specifies the dataset. Please choose among `amazon`, `youtube`, `dblp`, and `livejournal`. 
- `--model` specifies the gnn model. We mainly used `sign` in our experiments. 
- `--task_idxes` specifies the indexes of tasks that the model is trained on. Use the numbers from `0`  up to the number of tasks. Use space in between the indexes. 
- `--save_name` specifies the filename that saves the training results. Specify a name for the file, if one is going to use the results later.  

We show an example below that runs a SIGN model on the youtube dataset: 

```python
python train_multitask.py --dataset youtube --feature_dim 128\
    --model sign --num_layers 3 --hidden_channels 256 --lr 0.01 --dropout 0.1 --mlp_layers 2\
    --evaluator f1_score --sample_method decoupling --batch_size 1000 --epochs 100 --device 2 --runs 3\
    --save_name test --task_idxes 0 1 2 3 4 

```

<br/>

Use `train_sample_tasks.py` for sampling tasks and evaluating MTL performance on the trained models. Please specify the following key parameters.

-  `--dataset` specifies the dataset. Please choose among `amazon`, `youtube`, `dblp`, and `livejournal`. 
- `--num_samples` specifies the number of sampled subsets.
- `--min_task_num` specifies the minimum number of tasks in a subset.
- `--max_task_num` specifies the maximum number of tasks in a subset. 
- `--task_set_name` specifies the name of the file used for saving the sampled subsets. 
- `--save_name` specifies the filename that saves the training results. 

We show an example below that conducts the sampling process on the youtube dataset: 

```python
python train_sample_tasks.py --dataset youtube\
    --model sign --num_layers 3 --hidden_channels 256 --lr 0.01 --dropout 0.1\
    --evaluator f1_score --sample_method decoupling --batch_size 1000 --epochs 100 --device 2 --runs 1\
    --target_tasks none --num_samples 2000 --min_task_num 5 --max_task_num 5\
    --task_set_name sample_youtube --save_name sample_youtube
```

After sampling, we estimate the task affinities from the results and conduct clustering on task affinities to obtain task groups. We show a script to cluster tasks under `./notebooks/run_task_grouping.py`. 

<br/>

### **Section 2: Molecule property prediction**

Use `train_multitask.py` and change the `--dataset` to `alchemy_full`, `QM9`, or `molpcba`. The other parameters follow the ones used in community detection.

We show an example below to launch experiments on the alchemy, QM9, and ogb-molpcba datasets. 

```python
python train_multitask.py --dataset alchemy_full --model gine\
        --criterion regression --evaluator mae --hidden_channels 64 \
        --epochs 200 --downsample 1.0\
        --device 0 --runs 3 \
        --save_name test --task_idx 0 1 2 3 4

python train_multitask.py --dataset QM9 --model gine\
        --criterion regression --evaluator mae --hidden_channels 64 \
        --epochs 200 --downsample 1.0\
        --device 0 --runs 3 \
        --save_name test --task_idx 0 1 2 3 4

python train_multitask.py --dataset molpcba --model gine\
    --criterion multilabel --evaluator precision --hidden_channels 300 \
    --epochs 100 --downsample 1.0 --batch_size 32\
    --device 1 --runs 3 --mnt_mode max --eval_separate\
    --save_name test --task_idx 0 1 2 3 4
```

<br/>

Use `train_sample_tasks.py` and change the `--dataset` to `alchemy_full`, `QM9`, or `molpcba`. The other parameters follow the ones used in community detection. For example: 

```
python train_sample_tasks.py --dataset alchemy_full\
    --epochs 20 --downsample 0.2 --device 3\
    --num_samples 200 --min_task_num 4 --max_task_num 4\
    --task_set_name sample_alchemy --save_name sample_alchemy
```

### Requirements

Please install the requirements before launching the experiments:

```
pip install -r requirements.txt
```

We list the key packages used in our code:

- `python>=3.6`
- `torch>=1.10.0`
- `torch-geometric>=2.0.3`
- `pytorch-lightning>1.5.10`
- `torchmetrics>=0.8.2`
- `ogb>=1.3.4`

### Citation

If you find this repository useful or happen to use it in a research paper, please cite our work with the following bib information.

```
@article{li2023boosting,
  title={Boosting Multitask Learning on Graphs through Higher-Order Task Affinities},
  author={Li, Dongyue and Ju, Haotian and Sharma, Aneesh and Zhang, Hongyang R},
  journal={SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2023}
}
```

### Ackonwledgement

Thanks to the authors of the following repositories for providing their implementation publicly available, which greatly helps us develop this code.

- [**Open-Graph-Benchmark**](https://github.com/snap-stanford/ogb)
- [**TUDatasets**](https://github.com/chrsmrrs/tudataset)

- [**Large-Scale-GCN-Benchmarking**](https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking)
