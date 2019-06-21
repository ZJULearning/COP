# COP: Customized Deep Model Compression via Regularized Correlation-Based Filter-Level Pruning
Code for IJCAI2019 paper: **COP: Customized Deep Model Compression via Regularized Correlation-Based Filter-Level Pruning**



## Requirements

Ubuntu 16.04

Python == 3.x

Tensorflow >= 1.10.0

Numpy >= 1.14.1

Older versions of dependencies are to be tested.


You are highly **recommended** to use our [docker image](https://github.com/cheerss/deep-docker), which satisfies all the requirements above.



## Usage
```
train_dir=vgg-cifar10-model
python train.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" --network="vgg16" # to train a model

python auto_prune.py --train_dir=$train_dir --dataset="cifar10" --data_dir="./data" --network="vgg16" --alpha=1.0 --beta=0.0 --gamma=3.0 --prune_rate=0.1 # to prune and finetune a pretrained model
```

Please see `train.sh` and `train_finetune.sh` for more usage




## Results







## Datasets and Architectures Supported

The following table shows the datasets and models which could be used directly now. You could also use other datasets and models with minor changes. Please see next section for how to experiments with your own datasets and models

|           | CIFAR10 | CIFAR100 | ImageNet |
| --------- | ------- | -------- | -------- |
| VGG11     | N       | N        | Y        |
| VGG16     | Y       | Y        | N        |
| ResNet    | Y       | Y        | Y        |
| MobileNet | Y       | Y        | Y        |
| ...       |         |          |          |



## Use Your Own Models and Datasets

### Add new dataset

1. You should put your new dataset in `./datasets/`, please see `./datasets/cifar10.py` for reference.

2. A dataset is obliged to parse the dataset file, do data augmentation, and batch the dataset

3. You need to implement at least 2 interfaces, they are `train_input_fn` and `test_input_fn`, one for training and the other for test

   - `train_input_fn` should take 3 parameters, which are "data_directory", "batch_size" and "epochs" respectively, and it should return a tf.data.Dataset-type class. You should also do shuffling, data augmentation and batch internally.
   - `test_input_fn` should take 2 parameters, which are  "data_directory" and "batch_size", and it should return a tf.data.Dataset-type class too.

4. You could also define other functions you need.

5. Import the new dataset to `config.py` and add the new term to `parse_net_and_dataset` in `config.py` (just mimic what CIFAR10 does).

   

Totally speaking, you should create a new file `datasets/new_dataset.py` which contains at least the following 2 functions and tell `config.py` to parse the dataset correctly. 

```
def train_input_fn(data_dir, batch_size, epochs, **kargs):
	"""
	Args:
    data_dir: the path of the dataset file
    batch_size: the batch size of the dataset
    epochs: the dataset could provide how many epochs, -1 for infinity
    **kargs: any other parameters you want
  Return:
  	dataset: an object of type tf.data.Dataset
	"""
	pass

def test_input_fn(data_dir, batch_size, **kargs):
	"""
	Args:
    data_dir: the path of the dataset file
    batch_size: the batch size of the dataset
    **kargs: any other parameters you want
  Return:
  	dataset: an object of type tf.data.Dataset
	"""
	pass
```



### Add new model

1. You need to create 2 new files to use a  new model, one for the definition of the model and the other for the pruning details of the model. Put the first file in `./networks/` and the second file in `./prune_algorithm/`.
2. For the definition of the model, you should inherit from the class `ClassificationBase`, it has implement some essential functions for you. You only need to implement all abstract methods defined in `ClassificationBase`. See the comments in `ClassificationBase` for details. (You could see `./networks/vgg16` for reference).
3. For the pruning algorithm of the model, you should inherit from the class `PruneBase`, you need to implement all abstract methods defined in `PruneBase` and overload other methods if needed. See the comments in `PruneBase` for details. (You could see `./prune_algorithm/prune_vgg16` for reference).
4. Import the new model to `config.py` and parse the model correctly.

