# Kincannon Wilson Spring '23 Research

This is the repo for my research during Spring 2023.
It's currently a work in progress, and the contained
here is entirely experimental. The research involves
using CLIP and SVMs to identify multiple failure 
directions for a trained classifier. Feel free to 
shoot me an email at kgwilson2@wisc.edu if 
you're curious and want to learn more.

Research originally inspired by 
[this paper](https://gradientscience.org/failure-directions/) 
from the Madry Lab at MIT. Advised by 
[Dr. Yong Jae Lee](https://pages.cs.wisc.edu/~yongjaelee/)
and [Utkarsh Ojha](https://utkarshojha.github.io/).

---

## Overview

Deep learning models often exhibit consistent patterns of error,
often failing on hard sub-populations in the data. For instance, 
in a dataset where old men and young women are over-represented,
a binrary age classifier (with output 'old' or 'young') 
trained on such a dataset will likely rely on this 
false correlation to make its judgements.

My research work in the Spring '23 semester involved recreating, 
from scratch, the research produced by the Madry lab in the 
paper linked above. This research involves training an SVM
(for each class, such as 'old' or 'young' in the data) on the 
CLIP embedding space to predict which images the original
classifier got correct or incorrect. Because the SVM generates
a decision boundary, the normal vector to the boundary 
represents the primary failure direction of the model. For instance,
in our age classifier example, for the SVM trained on the embeddings 
for images in the 'old' class, we would expect more females on the
'incorrect' side as we get further from the boundary and more males 
on the 'correct' side as we get further away from the boundary.

Next, I began experimenting with more
complicated combinations of technologies such as SVMs plus
GMMs (Gaussian Mixture Models) in an effort to make the 
description of failure directions more robust and in order
to detect multiple failure directions of a classifier. 
For example, such an approach could detect that the classifier
learned a false correlation between the target classes 
(old/young) and sex (male/female) as well as a correlation
between the target classes and an additional factor such as 
facial pose (smiling/not smiling).

As my first foray into graduate-level research, this semester 
taught me so much! In particular, I learned more about the following 
technologies:

- PyTorch
- ResNet
- SVMs (Support Vector Machines)
- GMMs (Gaussian Mixture Models)
- CLIP (OpenAI's Contrastive Language-Image Pre-training model)

Please check out some of the contents of the "results" folder 
in this repo to see some figures illustrating my results. 
Feel free to poke around the rest of the code base and 
please shoot me an email at CannonGWilson@gmail.com 
if you have any questions!

---

## Setup

This setup process assumes python is downloaded.
It also assumes you're running this on a Linux system
with a GPU.

Make sure pip is installed & up to date, then install venv
```
python3 -m ensurepip --upgrade
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```

Create/activate the env
```
python3 -m venv research-env
. research-env/bin/activate
```

Install packages into env
```
pip install -r requirements.txt
```

Install CelebA zip file locally
(hosted on my Google Drive)
```
gdown "https://drive.google.com/uc?id=1HpoLLP9x7ON5nn5TnC7CPf2uoRnO2VnD"
```

Unzip the CelebA zip file
```
unzip img_align_celeba.zip
```

Create and fill the train/val/test directories
```
python3 dataset_utils/make_celeba_split.py
```

Before continuing, make sure the environment
variables stored in settings.py match the actual 
paths of your device. Then, try running 
the other files such as `test_evals.py`.

---

## Experiments

This project contains the various
experiments I've conducted so 
far. 

The test_evals.py file will 
evaluate the chosen model
(set in settings.py) on the 
test set, keeping track 
of the model's performance
on each subgroup.

The top_k.py file will generate
an SVM decision score for each test
image and use those scores to order
the test images. This experiment 
shows whether or not the SVM 
decision score can be used to 
surface failure directions in the
CLIP latent space caused by the 
classifier relying on a spurious
correlation to make its decisions.
I find that ordering by decision score
surfaces members of the minority 
subgroup for the strongest correlation
(sex and age) but is much less effective
for the second correlation (smiling
and age).

The gmm.py file represents the current 
line of inquiry for this project. 
The hope is that by using 
Guassian Mixture Models (GMMs) in 
combination with SVMs, our approach 
will be able to combine the 
interpretability of SVMs with the 
ability of GMMs to identify 
multiple subgroups. This reasoning
is inspired by [Domino](https://github.com/HazyResearch/domino). 
