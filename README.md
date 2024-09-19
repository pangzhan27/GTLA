# Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment

This repository is the official implementation of our paper "Long-Tail Temporal Action Segmentation via Group-wise Temporal Logit Adjustment" with model MSTCN on Breakfast dataset.

## Dataset
The dataset used in our paper is open-source data. It can be downloaded from the references in the main paper. 

## Training
To train the models in the paper, run this command:

```train
python activity_weighted_tla.py --action train --split 1 --seed 42 --w 0.5 --tau 0.5
```

You can find the scripts for other long-tailed methods in the code folder.

## Evaluation

For evaluation, you need first generate the prediction for test set. For example,
```eval
python activity_weighted_tla.py --action predict --split 1 --seed 42 --w 0.5 --tau 0.5
```
Then, use the 'eval.py' script for evaluation. The results contain both global and balanced metrics.

```eval
python eval.py --method prediction_path
```




 
