import os
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from mean_train import MEANS, STDEVS

load_dotenv()

# Vars
CALC_SVM_ACC = True
NUM_CORRS = int(os.getenv("NUM_CORRS"))
assert NUM_CORRS in [1,2], \
    "Only 1 or 2 correlations currently supported."
BATCH_SIZE = 512

print('Initializing Models and Loaders')

# Misc vars
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODES = ["old", "young"]
sigmoid = nn.Sigmoid()

# Custom model
CUSTOM_MODEL_PATH = os.getenv("MODEL_PATH")
custom_model = torchvision.models.resnet18()
custom_model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
custom_model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=DEVICE))
custom_model.to(DEVICE)

img_size = (int(os.getenv("IMG_WIDTH")), int(os.getenv("IMG_HEIGHT")))
assert img_size == (75, 75), "Images must be 75x75"
data_transforms = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=list(MEANS.values()), std=list(STDEVS.values()))
])

# SVMs are trained on *val* set
# Top_K is evaluated on *test* set
VAL_DIR, TEST_DIR = os.getenv("VAL_DIR"), os.getenv("TEST_DIR")
# val_data = datasets.ImageFolder(VAL_DIR, transform=data_transforms)
# val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
val_loader_no_trans = DataLoader(datasets.ImageFolder(VAL_DIR), batch_size=1)
NUM_VAL_IMGS = len(val_loader_no_trans)
test_data = datasets.ImageFolder(TEST_DIR, transform=data_transforms)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
test_loader_no_trans = DataLoader(datasets.ImageFolder(TEST_DIR), batch_size=1)
NUM_TEST_IMGS = len(test_loader_no_trans)
trained_svms, scalers = [], []

# CLIP
EMBEDDING_DIM = 512
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

for mode in MODES:

    current_class_num = 1 if mode == 'young' else 0
    paths = [tup[0] for tup in val_loader_no_trans.dataset.samples \
                if tup[1] == current_class_num]
    num_imgs_this_class = len(paths)

    with torch.no_grad():
        print(f'Finding model correctness for {mode.upper()} val images')
        custom_model.eval()
        correctness = torch.empty(num_imgs_this_class, dtype=torch.int8, device=DEVICE)
        cur_idx = 0
        cur_class_val_dir = os.path.join(VAL_DIR, mode) # ex: val/old
        cur_class_val_data = datasets.ImageFolder(cur_class_val_dir, transform=data_transforms)
        cur_class_val_loader = DataLoader(cur_class_val_data, batch_size=BATCH_SIZE)
        for i, (images, labels) in enumerate(cur_class_val_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            b_size = labels.size()[0]
            model_output = custom_model(images).squeeze()
            sig_out = sigmoid(model_output)
            preds = torch.where(sig_out>0.5, 1, 0)
            correct = torch.where(preds==labels, 1, -1)
            correctness[cur_idx:cur_idx+b_size] = correct
            cur_idx += b_size
        del cur_class_val_data
        del cur_class_val_loader

    with torch.no_grad():
        print(f'Getting CLIP embeddings for {mode.upper()} val images')
        clip_model.eval()
        pil_images = []
        for path in paths: 
            pil_images.append(clip_preprocess(Image.open(path)))
        image_input = torch.tensor(np.stack(pil_images), device=DEVICE)
        img_feature_stack = clip_model.encode_image(image_input).float() # [2000, 512]
        print(f"Expected {(num_imgs_this_class, EMBEDDING_DIM)}, got {img_feature_stack.size()}")

    print('Finished getting clip embeddings and correctness scores.')
    print('Beginning to fit SVM classifier for class ', mode)
    svm_classifier = LinearSVC(max_iter=5000) # old code: svm_classifier = svm.SVC(kernel="linear")
    np_feat_stack = img_feature_stack.cpu().numpy()
    np_corr = np.array(correctness.cpu(), dtype=np.int8)
    scaler = StandardScaler()
    scaler.fit(np_feat_stack)
    scaled_feat_stack = scaler.transform(np_feat_stack)
    svm_classifier.fit(scaled_feat_stack, np_corr)
    trained_svms.append(svm_classifier)
    scalers.append(scaler)

assert len(MODES) == len(trained_svms) == len(scalers), \
    "Number of fitted SVMs not equal to number of classes"

print("Finished training SVMs on validation data.")
for mode, svm_c, scaler in zip(MODES, trained_svms, scalers):

    # Calculate SVM accuracy
    #  -- Loop over all test images in class
    #  -- Get correctness
    #  -- Compare SVM output with correctness to get svm acc

    current_class_num = 1 if mode == 'young' else 0
    test_paths = [tup[0] for tup in test_loader_no_trans.dataset.samples \
                    if tup[1] == current_class_num]
    IMGS_THIS_CLASS = len(test_paths)
    confidences = torch.empty(IMGS_THIS_CLASS)
    if CALC_SVM_ACC:
        test_correctness = torch.empty(IMGS_THIS_CLASS)
    ds_values = None
    pil_images, sexes = [], np.empty(IMGS_THIS_CLASS)
    smiles = np.empty(IMGS_THIS_CLASS) if NUM_CORRS == 2 else None

    with torch.no_grad():
        print('Calculating model confidences for test images in class ', mode)
        cur_idx = 0
        cur_class_test_dir = os.path.join(TEST_DIR, mode) # ex: test/old
        cur_class_test_data = datasets.ImageFolder(cur_class_test_dir, transform=data_transforms)
        cur_class_test_loader = DataLoader(cur_class_test_data, batch_size=BATCH_SIZE)
        for i, (images, labels) in enumerate(cur_class_test_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            b_size = labels.size()[0]
            model_output = custom_model(images).squeeze()
            sig_out = sigmoid(model_output)
            conf = torch.abs(0.5-sig_out) # confidence represented by distance to 0.5
            confidences[cur_idx:cur_idx+b_size] = conf
            if CALC_SVM_ACC:
                preds = torch.where(sig_out>0.5, 1, 0)
                correct = torch.where(preds==labels, 1, -1)
                test_correctness[cur_idx:cur_idx+b_size] = correct
            cur_idx += b_size
    
    with torch.no_grad():
        print("Getting CLIP embeddings, attributes, and decision scores " +\
                "for test images in class ", mode)
        for i, path in enumerate(test_paths): 
            pil_images.append(clip_preprocess(Image.open(path)))
            # Also record sex/smiling of this test image
            sexes[i] = 0 if 'female' in path else 1
            if NUM_CORRS == 2:
                smiles[i] = 0 if 'no_smile' in path else 1
        image_input = torch.tensor(np.stack(pil_images), device=DEVICE)
        test_feat_stack = clip_model.encode_image(image_input).float()
        print(f"Test feature stack size: {test_feat_stack.size()}")
        scaled_test_feats = scaler.transform(test_feat_stack)
        ds_values = svm_c.decision_function(scaled_test_feats)

    if CALC_SVM_ACC:
        test_correctness = test_correctness.cpu().numpy()
        ds_correctness = np.where(ds_values >= 0, 1, -1) # equivalent to np.sign but no possibility of 0s
        total = len(test_correctness)
        corr = (test_correctness == ds_correctness).sum()
        print(f"SVM accuracy for class {mode}: {corr/total}")

    print('Plotting/saving results for class ', mode)
    conf_sorted_idxs =  np.argsort(confidences.cpu().numpy())
    ds_sorted_idxs = np.argsort(ds_values)
    conf_sorted_sexes = sexes[conf_sorted_idxs]
    ds_sorted_sexes = sexes[ds_sorted_idxs]
    conf_sorted_frac_male = np.empty(IMGS_THIS_CLASS)
    ds_sorted_frac_male = np.empty(IMGS_THIS_CLASS)
    if NUM_CORRS == 2:
        conf_sorted_smiles = smiles[conf_sorted_idxs]
        ds_sorted_smiles = smiles[ds_sorted_idxs]
        conf_sorted_frac_smiles = np.empty(IMGS_THIS_CLASS)
        ds_sorted_frac_smiles = np.empty(IMGS_THIS_CLASS)

    for num_people in range(1, IMGS_THIS_CLASS+1):
        conf_num_males = conf_sorted_sexes[:num_people].sum()
        ds_num_males = ds_sorted_sexes[:num_people].sum()
        conf_sorted_frac_male[num_people-1] = conf_num_males / num_people
        ds_sorted_frac_male[num_people-1] = ds_num_males / num_people
        if NUM_CORRS == 2:
            conf_num_smiles = conf_sorted_smiles[:num_people].sum()
            ds_num_smiles = ds_sorted_smiles[:num_people].sum()
            conf_sorted_frac_smiles[num_people] = conf_num_smiles / num_people
            ds_sorted_frac_smiles[num_people] = ds_num_smiles / num_people

    if mode == "old":
        minority_sex = "Female" # pylint: disable=invalid-name
        sex_y_conf = 1-conf_sorted_frac_male
        sex_y_ds = 1-ds_sorted_frac_male
        sex_baseline = 1 - (conf_num_males / IMGS_THIS_CLASS)
        if NUM_CORRS == 2:
            minority_smile = "Smiling"
            smi_y_conf = conf_sorted_frac_smiles
            smi_y_ds = ds_sorted_frac_smiles
            smi_baseline = conf_num_smiles / IMGS_THIS_CLASS

    elif mode == "young":
        minority_sex = "Male" # pylint: disable=invalid-name
        sex_y_conf = conf_sorted_frac_male
        sex_y_ds = ds_sorted_frac_male
        sex_baseline = conf_num_males / IMGS_THIS_CLASS
        if NUM_CORRS == 2:
            minority_smile = "Not Smiling"
            smi_y_conf = 1-conf_sorted_frac_smiles
            smi_y_ds = 1-ds_sorted_frac_smiles
            smi_baseline = 1 - (conf_num_smiles / IMGS_THIS_CLASS)

    # Plot sex results for class
    plt.plot(range(IMGS_THIS_CLASS), sex_y_conf, color='g', label="Confidence")
    plt.plot(range(IMGS_THIS_CLASS), sex_y_ds, color='b', label="Decision Score")
    plt.axhline(y=sex_baseline, color='r', label="Baseline")
    plt.ylabel(f'Fraction {minority_sex}')
    plt.xlabel("Top K Flagged")
    plt.legend(loc="upper right")
    plt.title(f"{minority_sex} Flagged for Class {mode}")
    plt.savefig(f'new_{mode}_orig_results.png')
    plt.show()
    plt.clf()
    plt.close()

    # Plot smiling results for class if needed
    if NUM_CORRS == 2:
        # Plot smiling results for class
        plt.plot(range(IMGS_THIS_CLASS), smi_y_conf, color='g', label="Confidence")
        plt.plot(range(IMGS_THIS_CLASS), smi_y_ds, color='b', label="Decision Score")
        plt.axhline(y=smi_baseline, color='r', label="Baseline")
        plt.ylabel(f'Fraction {minority_smile}')
        plt.xlabel("Top K Flagged")
        plt.legend(loc="upper right")
        plt.title(f"{minority_smile} Flagged for Class {mode}")
        plt.show()
        plt.clf()
        plt.close()
