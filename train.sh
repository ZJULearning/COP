## vgg-cifar10
train_dir="vgg-cifar10-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=0 python -u train.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" \
    --network="vgg16" 2>&1 | tee -a $train_dir/train.log

## vgg-cifar100
train_dir="vgg-cifar100-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=1 python -u train.py --train_dir=$train_dir --dataset="cifar100" --data_dir="./data" \
    --network="vgg16" 2>&1 | tee -a $train_dir/train.log

##vgg11-imagenet
train_dir="vgg-imagenet-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=4,5 python -u train.py --train_dir=$train_dir --dataset="imagenet" --data_dir="./data" \
    --network="vgg11" 2>&1 | tee -a $train_dir/train.log

## mobilenet-cifar10
train_dir="mobilenet-cifar10-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=2 python -u train.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" \
    --network="mobilenet_for_cifar" 2>&1 | tee -a $train_dir/train.log

## mobilenet-cifar100
train_dir="mobilenet-cifar100-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=2 python -u train.py --train_dir=$train_dir --dataset="cifar100" --data_dir="./data" \
    --network="mobilenet_for_cifar" 2>&1 | tee -a $train_dir/train.log

## mobilenet-imagenet
train_dir="mobilenet-imagenet-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=6,7 python -u train.py --train_dir=$train_dir --dataset="imagenet" --data_dir="./data" \
    --network="mobilenet_for_imagenet" 2>&1 | tee -a $train_dir/train.log

## resnet-cifar10
train_dir="resnet32-cifar10-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=3 python -u train.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" \
    --network="resnet32" 2>&1 | tee -a $train_dir/train.log

## resnet-cifar100
train_dir="resnet32-cifar100-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=7 python -u train.py --train_dir=$train_dir --dataset="cifar100" --data_dir="./data" \
    --network="resnet32" 2>&1 | tee -a $train_dir/train.log

## resnet-imagenet
train_dir="resnet18-imagenet-model/n1"
mkdir -p $train_dir
CUDA_VISIBLE_DEVICES=1,2 python -u train.py --train_dir=$train_dir --dataset="imagenet" --data_dir="./data" \
    --network="resnet18" 2>&1 | tee -a $train_dir/train.log
