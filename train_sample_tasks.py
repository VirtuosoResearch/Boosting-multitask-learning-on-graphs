import os 
import gc
import argparse
import numpy as np

name_to_num_classes = {
    "youtube": 100,
    "dblp": 100,
    "amazon": 100,
    "livejournal": 100,
    "alchemy_full": 12,
    "QM9": 12,
    "molpcba": 128
}

def main(args):
    task_num = name_to_num_classes[args.dataset]
    task_list = [str(i) for i in range(task_num)]
    target_tasks = args.target_tasks
    other_tasks = [task for task in task_list if task not in target_tasks]

    num_samples = args.num_samples
    max_task_num = args.max_task_num
    min_task_num = args.min_task_num
    for _ in range(num_samples):
        # create a set of trained task combinations
        sampled_task_dir = os.path.join("./sampled_tasks", "{}.txt".format(args.task_set_name))
        if not os.path.exists(sampled_task_dir):
            f = open(sampled_task_dir, "w")
            f.close()
            
        with open(sampled_task_dir, "r") as f:
            sampled_tasks = set()
            for line in f.readlines():
                sampled_tasks.add(line.rstrip("\n"))
            # print(sampled_tasks)

        # train on a new task combination
        with open(sampled_task_dir, "a") as f:
            if target_tasks[0] == "none":
                tmp_other_task_num = np.random.randint(
                    low=min_task_num, high=max_task_num+1
                )
                tmp_sampled_other_tasks = np.random.choice(other_tasks, size=tmp_other_task_num,replace=False)
                
                tmp_sampled_tasks = tmp_sampled_other_tasks
                tmp_sampled_tasks.sort()
                tmp_sampled_tasks = " ".join(tmp_sampled_tasks)
            else:
                tmp_target_task_num = np.random.randint(low=1, high=len(target_tasks)+1)
                tmp_sampled_target_tasks = np.random.choice(target_tasks, size=tmp_target_task_num, replace=False)

                tmp_other_task_num = np.random.randint(
                    low=max(min_task_num-tmp_target_task_num, 0), 
                    high=max_task_num-tmp_target_task_num+1
                )
                tmp_sampled_other_tasks = np.random.choice(other_tasks, size=tmp_other_task_num,replace=False)
                
                tmp_sampled_tasks = np.concatenate([tmp_sampled_target_tasks, tmp_sampled_other_tasks])
                tmp_sampled_tasks.sort()
                tmp_sampled_tasks = " ".join(tmp_sampled_tasks)
            
            if tmp_sampled_tasks in sampled_tasks:
                continue
            print(tmp_sampled_tasks)
            
            if args.dataset == "molpcba":
                run_str = "--criterion multilabel --evaluator precision --hidden_channels 300 --batch_size 32 --mnt_mode max --eval_separate"
                os.system("python train_multitask.py --dataset {} --task_idxes {}\
                        --epochs {} --downsample {} --model gine\
                        --device {} --runs {} --save_name {} {}".format(
                        args.dataset, tmp_sampled_tasks, 
                        args.epochs, args.downsample,
                        args.device, args.runs, args.save_name, run_str
                ))
            elif args.dataset == "alchemy_full" or args.dataset == "QM9":
                run_str = ""
                os.system("python train_multitask.py --dataset {} --task_idxes {}\
                        --epochs {} --downsample {} --criterion regression --evaluator mae --model gine\
                        --device {} --runs {} --save_name {} {}".format(
                        args.dataset, tmp_sampled_tasks, 
                        args.epochs, args.downsample,
                        args.device, args.runs, args.save_name, run_str
                ))
            else:
                os.system("python train_multitask.py --dataset {} --num_communities {} --task_idxes {}\
                        --model {} --num_layers {} --hidden_channels {} --dropout {} --mlp_layers 2 --lr {}\
                        --evaluator {} --sample_method {} --batch_size {} --epochs {} --downsample {}\
                        --device {} --runs {} --save_name {}".format(
                        args.dataset, args.num_communities, tmp_sampled_tasks, 
                        args.model, args.num_layers, args.hidden_channels, args.dropout, args.lr,
                        args.evaluator, args.sample_method, args.batch_size, args.epochs, args.downsample,
                        args.device, args.runs, args.save_name
                ))
            gc.collect()
            sampled_tasks.add(tmp_sampled_tasks)
            f.write(tmp_sampled_tasks + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--model', type=str, default='sign')
    parser.add_argument("--num_communities", type=int, default=100)
    # Model
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    # Training
    parser.add_argument('--evaluator', type=str, default="f1_score")
    parser.add_argument('--sample_method', type=str, default="decoupling")
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument('--downsample', type=float, default=1.0)
    
    # Sample tasks
    parser.add_argument("--target_tasks", nargs='+', type=str, default=["none"])
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--min_task_num", type=int, default=10)
    parser.add_argument("--max_task_num", type=int, default=10)
    parser.add_argument("--task_set_name", type=str, default="sampled_tasks")
    parser.add_argument("--save_name", type=str, default="sampled_tasks")
    args = parser.parse_args()
    main(args)