All findings can be verified
by running the respective files.

## Output from test_evals.py:

Both models achieved ~100%
accuracy on every subgroup in 
the training data. Both models
performed significantly better
on the more populous subgroups,
suggesting that they relied 
on the planted correlations
to aid their decision-making.

1 Corr Dataset/Model:

| Subgroup | Val Accuracy | Test Accuracy |
| --- | --- | --- |
| Total | 0.83 | 0.86 |
| OLD_MALE | 0.98 | 0.98 |
| OLD_FEMALE | 0.78 | 0.76 |
| YOUNG_MALE | 0.62 | 0.64 |
| YOUNG_FEMALE | 0.93 | 0.94 |

2 Corr Dataset/Model:

| Subgroup | Val Accuracy | Test Accuracy |
| --- | --- | --- |
| Total | 0.83 | 0.80 |
| OLD_MALE_SMILE | 0.95 | 0.95 |
| OLD_MALE_NO_SMILE | 0.99 | 0.98 |
| OLD_FEMALE_SMILE | 0.57 | 0.59 |
| OLD_FEMALE_NO_SMILE | 0.97 | 0.95 |
| YOUNG_MALE_SMILE | 0.77 | 0.74 |
| YOUNG_MALE_NO_SMILE | 0.5 | 0.51 |
| YOUNG_FEMALE_SMILE | 0.96 | 0.96 |
| YOUNG_FEMALE_NO_SMILE | 0.89 | 0.9 |



## Output from top_k.py:

1 Corr Dataset/Model:

SVM accuracy for class old: 0.8667491749174917

SVM accuracy for class young: 0.8305544528251951


2 Corr Dataset/Model:

SVM accuracy for class old: 0.7283090563647878

SVM accuracy for class young: 0.8060874655071494

## Findings from gmm.py:

You can read more about the metric I came up 
with to assess the quality of a cluster/path
between clusters in the docstring at the
top of gmm.py.

(H)DBSCAN* < KMeans < AgglomerativeClustering < GMM

*Density-based clustering methods (DBSCAN and HDBSCAN)
did not work at all in the CLIP embedding space.
There was no combination of hyperparameters 
(epsilon/min_samples) that could separating the 
embeddings in this space.

## Thoughts:

Takeaways so far:
The SVM does a good job of surfacing the primary
failure direction/primary correlation in the 
datasets but does not do well at finding 
the secondary correlation.

Misc findings:
Scaling the clip embeddings before training/
predicting the SVM reduced performance.
sklearn.svm.SVC does better than 
sklearn.svm.LinearSVC.
Clustering methods ordered by effectiveness as
found in gmm

Other thoughts: 
The CLIP embedding space is capable of encoding
many different subjects in images, while we're
only using it on images of faces in these
experiments so far. Dimensionality reduction
on the calculated embeddings using PCA would
likely speed up the experiments without 
sacrificing accuracy.
