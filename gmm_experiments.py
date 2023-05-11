import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import os
import copy
from functools import reduce
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import distance
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from mean_train import MEANS, STDEVS

load_dotenv()

print('Initializing Models')

# Misc vars
SHOW_IMGS = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODES = ['young'] # ["old", "young"]
BATCH_SIZE = 512
OUT_FEATS = 2
NUM_COMPS = 4 # number of subgroups on each side of the dividing hyperplane
NUM_IMGS = 200 # max number of images to include in animate
sigmoid = nn.Sigmoid()

# Custom model
CUSTOM_MODEL_PATH = os.getenv("MODEL_PATH")
custom_model = torchvision.models.resnet18()
custom_model.fc = nn.Linear(in_features=512, out_features=OUT_FEATS, bias=True)
custom_model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=DEVICE))
custom_model.to(DEVICE)

img_size = (int(os.getenv("IMG_WIDTH")), int(os.getenv("IMG_HEIGHT")))
assert img_size == (75, 75), "Images must be 75x75"
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(MEANS.values()), std=list(STDEVS.values()))
])

# Loaders
VAL_DIR, TEST_DIR = os.getenv("VAL_DIR"), os.getenv("TEST_DIR")
val_loader_no_trans = DataLoader(datasets.ImageFolder(VAL_DIR), batch_size=1)
NUM_VAL_IMGS = len(val_loader_no_trans)
test_loader_no_trans = DataLoader(datasets.ImageFolder(TEST_DIR), batch_size=1)

# CLIP
EMBEDDING_DIM = 512
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

svms = []

for mode in MODES:

    current_class_num = 1 if mode == 'young' else 0
    paths = [tup[0] for tup in val_loader_no_trans.dataset.samples \
                if tup[1] == current_class_num]
    num_imgs_this_class = len(paths)

    cur_class_val_dir = os.path.join(VAL_DIR, mode) # ex: val/old
    cur_class_val_data = datasets.ImageFolder(cur_class_val_dir, transform=data_transforms)
    cur_class_val_loader = DataLoader(cur_class_val_data, batch_size=BATCH_SIZE)

    # Age Classifier Correctness
    print("Getting correctness for class ", mode)
    correctness = torch.empty(num_imgs_this_class, device=DEVICE, dtype=torch.int8)
    with torch.no_grad():
        for i, (images, labels) in enumerate(cur_class_val_loader):
            start_i = i * BATCH_SIZE
            end_i = start_i + labels.size()[0]
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            out = custom_model(images)
            preds = torch.argmax(out, dim=1)
            corr = preds == labels
            correctness[start_i:end_i] = corr

    # CLIP 
    print("Getting clip embeds for class ", mode)
    with torch.no_grad():
        clip_model.eval()
        pil_images = []
        for path in paths: 
            pil_images.append(clip_preprocess(Image.open(path)))
        image_input = torch.tensor(np.stack(pil_images), device=DEVICE)
        clip_embeds = torch.empty((num_imgs_this_class, EMBEDDING_DIM), device=DEVICE)
        for start_i in range(0, num_imgs_this_class, BATCH_SIZE):
            end_i = start_i + BATCH_SIZE if start_i + BATCH_SIZE < num_imgs_this_class \
                else num_imgs_this_class
            clip_embeds[start_i:end_i] = clip_model.encode_image(image_input[start_i:end_i]).float()

    # Train SVM
    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(clip_embeds, correctness)
    svms.append(svm_classifier)

    # Find CLIP embeddings above/below
    # decision boundary
    ds_values = np.dot(svm_classifier.coef_[0], \
            clip_embeds.cpu().numpy().transpose()) + \
                svm_classifier.intercept_[0]
    easy_idxs = np.where(ds_values >= 0)[0]
    diff_idxs = np.where(ds_values < 0)[0]
    easy_gm = GaussianMixture(n_components=NUM_COMPS, random_state=0).fit(clip_embeds[easy_idxs])
    diff_gm = GaussianMixture(n_components=NUM_COMPS, random_state=0).fit(clip_embeds[diff_idxs])

    # Loop over each combination of easy-diff centers
    anim_paths = {}
    for easy_i in range(NUM_COMPS):
        for diff_i in range(NUM_COMPS):
            cur_key = f"{easy_i}-{diff_i}"
            anim_paths[cur_key] = []
            start_pt = easy_gm.means_[easy_i]
            end_pt = diff_gm.means_[diff_i]
            embed_path = np.linspace(start_pt, end_pt, num=NUM_IMGS)
            for cur_pt in embed_path:
                # find the img with closest embedding to cur_pt
                min_dist = np.inf
                best_idx = None
                for idx, embed in enumerate(clip_embeds.cpu().numpy()):
                    dist = np.linalg.norm(cur_pt - embed)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                anim_paths[cur_key].append(best_idx)
            anim_paths[cur_key] = list(set(anim_paths[cur_key])) # remove non-unique path names

    # Show imgs
    if SHOW_IMGS:
        for key, lis in anim_paths.items():
            for img_idx in lis:
                pil_img = Image.open(paths[img_idx])
                plt.imshow(np.asarray(pil_img))
                plt.title(f"{mode} {key} {paths[img_idx]}")
                plt.show()
    """
    # Need some notion of how good the clusters are
    # in order to test different clustering methods
    # HDBSCAN vs GMM for instance
    # This would require measuring how well the cluster
    # captured ONLY females or males and ONLY smiles or
    # not smiles (smiling diff scores)

    # Then, need some notion of how good a path 
    # between clusters is. A path represents a 
    # failure direction if common atrributes 'flip',
    # i.e. there is a large change in the diff scores
    # computed above. Ex: cluster 0:
    # {'old_female_no_smile': 17,
    #  'old_female_smile': 5,
    #  'old_male_no_smile': 108,
    #  'old_male_smile': 60}
    # sex diff 0: 17+5 - (108+60) = -146
    # smiling diff 0: 17 + 108 - (5+60) = 60
    # compared with cluster 1:
    # {'old_female_no_smile': 88,
    #  'old_female_smile': 106,
    #  'old_male_no_smile': 1,
    #  'old_male_smile': 3}
    # sex diff 1: (88 + 106) - (1+3) = 190
    # smiling diff 1: (88 + 1) - (106+3) = -20
    # Then, differences of diffs shows path quality:
    # sex diff = abs(-146 - 190) = 336
    # smiling diff = abs(60 - -20) = 80
    # Thus, the path from cluster 0 to cluster 1
    # has been scored on how well it captures the 
    # sex failure direction (very well) and
    # how well it captures the smiling failure 
    # direction (not as well) 
    """

    # Create NUM_COMP clusters
    # on the entire CLIP space, 
    # then compare by hard vs. easy
    # def test_acc(model, mode_desc):
    model = GaussianMixture(n_components=NUM_COMPS, random_state=0)
    mode_desc = 'Gaussian Mix'
    full_clusters = model.fit_predict(clip_embeds)
    full_res = {cluster: {f'{mode}_female_no_smile': 0,
                        f'{mode}_female_smile': 0,
                        f'{mode}_male_no_smile': 0,
                        f'{mode}_male_smile': 0} 
                        for cluster in range(NUM_COMPS)}
    for i, path in enumerate(paths):
        sex = 'female' if 'female' in path else 'male'
        smile = 'no_smile' if 'no_smile' in path else 'smile'
        cluster = full_clusters[i]
        key = f'{mode}_{"_".join([sex,smile])}'
        full_res[cluster][key] = full_res[cluster][key] + 1
    for cluster, res_dict in full_res.items():
        res_dict['sex_score'] = (res_dict[f'{mode}_female_no_smile'] + \
                                res_dict[f'{mode}_female_smile']) - \
                                (res_dict[f'{mode}_male_no_smile'] + \
                                res_dict[f'{mode}_male_smile'])                    
        res_dict['smile_score'] = (res_dict[f'{mode}_female_no_smile'] + \
                                res_dict[f'{mode}_male_no_smile']) - \
                                (res_dict[f'{mode}_female_smile'] + \
                                    res_dict[f'{mode}_male_smile'])
    all_pairings = [f"{e_i}-{d_i}" for e_i in range(NUM_COMPS) for d_i in range(NUM_COMPS)]
    full_diffs = {}
    for pair in all_pairings:
        e_i, d_i = pair.split("-")
        if f"{d_i}-{e_i}" in full_diffs or e_i == d_i:
            continue
        full_diffs[pair] = {'sex_diff':0, 'smile_diff':0}
    full_score = 0
    for pair, diff_dict in full_diffs.items():
        e_i, d_i = [int(i) for i in pair.split("-")]
        sex_diff = full_res[e_i]['sex_score'] - full_res[d_i]['sex_score']
        smile_diff = full_res[e_i]['smile_score'] - full_res[d_i]['smile_score']
        diff_dict['sex_diff'] = sex_diff
        diff_dict['smile_diff'] = smile_diff
        full_score += abs(sex_diff) + abs(smile_diff)

    print(f'Calculated abs score for mode: {mode_desc}', full_score)
    print('full_diffs: ', full_diffs)

    print("Now considering only hard vs. easy clusters")
    easy_clusters, diff_clusters = [], []
    for cluster_i in range(NUM_COMPS):
        mean = model.means_[cluster_i]
        score = np.dot(svm_classifier.coef_[0], \
            mean.transpose()) + \
            svm_classifier.intercept_[0]
        # TODO: smarter way to figure out which side 
        # should be considered easy/hard
        if (score >= 0 and mode == 'young') or \
            (score <=0 and mode == 'old'):
            easy_clusters.append(cluster_i)
        else:
            diff_clusters.append(cluster_i)

    mod_full_diffs = {}
    mod_score = 0
    print('finding clusters that cross DS boundary')
    for pair, pair_dic in full_diffs.items():
        e_i, d_i = [int(i) for i in pair.split("-")]
        if e_i in easy_clusters and d_i in diff_clusters:
            print(full_diffs[f'{e_i}-{d_i}'])
            mod_full_diffs[pair] = pair_dic
            mod_score += abs(pair_dic['sex_diff']) + \
                abs(pair_dic['smile_diff'])
    print('Difference score for only clusters that ' +\
          f'cross decision boundary: {mod_score}')

    # test_acc(GaussianMixture(n_components=NUM_COMPS, random_state=0), 'Gaussian Mix')
    # test_acc(KMeans(n_clusters = NUM_COMPS,  random_state = 0), 'KMeans')
    # test_acc(AgglomerativeClustering(n_clusters = NUM_COMPS), 'Graphical Clustering')
    # test_acc(DBSCAN(eps=0.8, min_samples=50), 'DBSCAN' )
