# Reimplementation_SSC
Pytorch unofficial reimplementation for "Spatial and semantic consistency regularizations for pedestrian attribute recognition", ICCV2021

### Results 

|      PETA      |   mA  |  Acc  |  Prec | Recall |   F1  |
|:--------------:|:-----:|:-----:|:-----:|:------:|:-----:|
|  Paper SSC  | 86.52 | 78.95 | 86.02 |  87.12 | 86.99 |
| Unofficial SSC | 83.90 | 78.60 | 86.16 | 86.59  | 86.38 |

|     PA100K     |   mA  |  Acc  |  Prec | Recall |   F1  |
|:--------------:|:-----:|:-----:|:-----:|:------:|:-----:|
|  Paper SSC  | 81.87 | 78.89 | 85.98 |  89.10 | 86.87 |
| Unofficial SSC | 79.51 | 78.49 | 86.80 | 87.30  | 87.05 |

|       RAP      |   mA  |  Acc  |  Prec | Recall |   F1  |
|:--------------:|:-----:|:-----:|:-----:|:------:|:-----:|
|  Paper SSC  | 82.77 | 68.37 | 75.05 |  87.49 | 80.43 |
| Unofficial SSC | 82.14 | 68.16 | 77.87 | 82.88  | 79.87 |


### Data
You can download dataset from "https://github.com/valencebond/Rethinking_of_PAR".
   ```txt
   ${POSE_ROOT}
    |-- data
        |-- peta
            |-- images/
            |-- dataset_all.pkl
        |-- rap
            |-- RAP_dataset/
            |-- dataset_all.pkl
        |-- pa100k
            |-- data/
            |-- dataset_all.pkl
   ```

### Training
```bash
python tools/train.py --cfg experiments/[dataset name].yaml --gpu 0 -- savename [save path of outputs]
```


### Acknowledgements
Thanks for their open-source codes [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and [Jian Jia](https://github.com/valencebond/Rethinking_of_PAR).


