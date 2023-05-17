# Kincannon Wilson Spring '23 Research

This is the repo for my research during Spring 2023.
It's currently a work in progress. The research involves
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
python3 make_dataset.py
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
