## vgg-cifar10
prune_rate=0.67
train_dir="vgg-cifar10-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=1 python -u auto_prune.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" 
    --network="vgg16" --alpha=1.0 --beta=0.0 --gamma=3.0 --prune_rate=$prune_rate \
    2>&1 | tee -a $train_dir/$prune_rate/finetune.log

## vgg-cifar100
prune_rate=0.01
train_dir="vgg-cifar100-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=2 python -u auto_prune.py --train_dir=$train_dir --dataset="cifar100" --data_dir="./data" \
    --network="vgg16" --alpha=1.0 --beta=0.0 --gamma=3.0 --prune_rate=$prune_rate \
    2>&1 | tee -a $train_dir/$prune_rate/finetune.log

## vgg-imagenet
prune_rate=0.001
train_dir="vgg-imagenet-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=2,3 python -u auto_prune.py --train_dir=$train_dir --dataset="imagenet" --data_dir="./data" \
    --network="vgg11" --alpha=1.0 --beta=1.0 --gamma=1.0 --prune_rate=$prune_rate \
    2>&1 | tee -a $train_dir/$prune_rate/finetune.log

## mobilenet-cifar10
prune_rate=0.01
train_dir="mobilenet-cifar10-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=1 python -u auto_prune.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" \
    --network="mobilenet_for_cifar" 2>&1 | tee -a $train_dir/$prune_rate/finetune.log

## mobilenet-cifar100
prune_rate=0.01
train_dir="mobilenet-cifar100-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=1 python -u auto_prune.py --train_dir=$train_dir --dataset="cifar100" --data_dir="./data" \
    --network="mobilenet_for_cifar" 2>&1 | tee -a $train_dir/$prune_rate/finetune.log

## resnet32-cifar10
prune_rate=0.01
train_dir="resnet32-cifar10-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=3 python -u auto_prune.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" \
    --network="resnet32" 2>&1 | tee -a $train_dir/$prune_rate/finetune.log

## resnet32-cifar100
prune_rate=0.01
train_dir="resnet32-cifar100-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=3 python -u auto_prune.py --train_dir=$train_dir --dataset="cifar100" --data_dir="./data" \
    --network="resnet32" 2>&1 | tee -a $train_dir/$prune_rate/finetune.log

## resnet18-imagenet
prune_rate=0.001
train_dir="resnet18-imagenet-model/n1"
mkdir -p $train_dir/$prune_rate
CUDA_VISIBLE_DEVICES=1 python -u auto_prune.py --train_dir=$train_dir --dataset="imagenet" --data_dir="./data" \
    --network="resnet18" 2>&1 | tee -a $train_dir/$prune_rate/finetune.log