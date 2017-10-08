
The code of this repository was used for the following publication. If
you find this code useful please cite our paper:

```
@article{Gallego2017,
title = "Clustering-based k-Nearest Neighbor Classification for Large-Scale Data with Neural Codes Representation",
author = "Antonio-Javier Gallego and Jorge Calvo-Zaragoza and Jose J. Valero-Mas and Juan Ramón Rico-Juan",
journal = "Pattern Recognition",
year = "2017"
}
```

Below we include instructions for reproducing the experiments.
Basically, they consist in the following steps: preparing data,
training a neural network, exporting the neural codes, and finally
using the proposed ckNN+ algorithm.


## Datasets

Datasets must be stored as CSV-type text files, saving the tag first and then the features. Each dataset must also be divided into five files with the names `A.txt`, `B.txt`, `C.txt`, `D.txt`, and `E.txt`, corresponding to the five partitions that will be used for cross-validation. When creating these partitions it is recommended to shuffle the samples and stratify the distribution of samples of each class in each partition.

As example, the "USPS" dataset is included in the "`datasets/usps/original`" folder. This dataset can be downloaded publicly from https://www.csie.ntu.edu.tw/%7Ecjlin/libsvmtools/datasets/multiclass.html#usps



## Neural Network training

For each dataset, a neural network must be trained until obtaining an acceptable level of accuracy. Once the training is finished, the network will be used to export the features learned in the penultimate layer (also called Neural Codes). For example, to train a neural network for the USPS dataset, using the model 1 and the first fold as test and the rest as training, we will write:

```
$ python export_dnn.py -path datasets/usps/original -cv 0 -m 1 --save
```

This command starts the training and once completed it will store the Neural Codes in the folder "`datasets/usps/m1_cv0`". Features are saved as CSV files using the names "`train.txt`" and "`test.txt`", where "`train.txt`" will contain the data of all the folds used for the training.

This executable has several input parameters, for example, choose the network model (`-m`), the number of channels of the input data (`-c`), the type of preprocessing (`-pre`), or the "`--flat`" option when the input data is a 1D vector. These parameters can be consulted by typing the "`-h`" option or reading the source code directly.

Specifically, for the datasets of the publication, the following parameters were used:

| Dataset   | # model | Channels | Flat   | Preprocess |
| --------- | ------- | -------- | ------ | ---------- |
| usps      | 1       | 1        |        | None       |
| mnist     | 1       | 1        |        | 255        |
| gisette   | 3       | 1        | --flat | None       |
| letter    | 2       | 1        | --flat | None       |
| pendigits | 2       | 1        | --flat | None       |
| satimage  | 4       | 4        |        | None       |
| homus	    | 6       | 1        |        | 255        |
| nist	    | 6       | 1        |        | 255        |

For example, for satimage we execute "`python export_dnn.py -path datasets/satimage/original -cv 0 -m 4 -c 4 --save`" and for nist "`python export_dnn.py -path datasets/nist/original -cv 0 -m 6 -pre 255 --save`".



## ckNN+

Once the neural codes are obtained we may proceed to use the ckNN+ algorithm:

```
$ python cknn.py -path datasets/usps/m1_cv0
```

This command launches several experiments with the input data, using different number of partitions (10,15,20,25,30,100,500,1000) and different values of k for each of them (1,3,5,7,9).



## kNN

To get the results of kNN without creating any partition we can use the following command:

```
$ python knn.py -path datasets/usps/m1_cv0
```
