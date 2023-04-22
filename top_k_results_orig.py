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
from mean_train import MEANS, STDEVS

load_dotenv()

print('Initializing Models')

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

# Top_K is evaluated on *test* set
TEST_DIR = os.getenv("TEST_DIR")
test_data = datasets.ImageFolder(TEST_DIR, transform=data_transforms)
test_loader = DataLoader(test_data, batch_size=1)

# CLIP
EMBEDDING_DIM = 512
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

for mode in MODES:

    ds_values = []
    confidences = []
    sexes = []
    smiles = []
    correctness = []
    img_feature_stack = []

    print(f'Finding CLIP embeddings and attributes for {mode.upper()} images')
    with torch.no_grad():
        custom_model.eval()
        clip_model.eval()
        for i, (image, label) in enumerate(test_loader):

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            # Make sure image belongs to current class
            # being considered (young/old). Skip for now if not
            path = test_loader.dataset.samples[i][0] # ex: test/old/female/no_smile/000997.jpg
            img_class = path.split('/')[-4]
            if not mode == img_class:
                continue

            # Calculate correctness/confidence of custom model
            model_output = custom_model(image)
            sig_out = sigmoid(model_output).item()
            pred = 1 if sig_out > 0.5 else 0
            conf = abs(0.5 - sig_out)
            confidences.append(conf)
            actual = label.item()
            if pred == actual:
                correctness.append(1)
            else:
                correctness.append(-1)

            # Record sex/smiling of current img
            sex = path.split('/')[-3]
            sexes.append(1 if sex == "male" else 0)

            # Generate CLIP embedding
            pil_image = Image.open(path)
            clip_img = clip_preprocess(pil_image).unsqueeze(0).to(DEVICE)
            img_features = clip_model.encode_image(clip_img)
            img_feature_stack.append(img_features)

    print('Finished getting clip embeddings and correctness scores.')
    print('Beginning to fit SVM classifiers')
    # Fit the SVM on all of the embeddings
    IMGS_THIS_CLASS = len(correctness)
    svm_classifier = svm.SVC(kernel="linear")
    np_feat_stack = torch.cat(img_feature_stack).cpu().numpy()
    np_corr = np.array(correctness, dtype=np.int8)
    svm_classifier.fit(np_feat_stack, np_corr)
    for idx in range(IMGS_THIS_CLASS):
        score = np.dot(svm_classifier.coef_[0], \
            np_feat_stack[idx].transpose()) + \
                svm_classifier.intercept_
        ds_values.append(score[0])

    print('Fitted SVM, now plotting results.')
    conf_sorted_idxs =  np.argsort(confidences)
    ds_sorted_idxs = np.argsort(ds_values)
    conf_sorted_sexes = np.array(sexes)[conf_sorted_idxs]
    ds_sorted_sexes = np.array(sexes)[ds_sorted_idxs]
    conf_sorted_frac_male = np.empty(IMGS_THIS_CLASS)
    ds_sorted_frac_male = np.empty(IMGS_THIS_CLASS)

    for num_people in range(1, IMGS_THIS_CLASS+1):
        conf_num_males = conf_sorted_sexes[:num_people].sum()
        ds_num_males = ds_sorted_sexes[:num_people].sum()
        conf_sorted_frac_male[num_people-1] = conf_num_males / num_people
        ds_sorted_frac_male[num_people-1] = ds_num_males / num_people

    if mode == "old":
        minority_sex = "Female" # pylint: disable=invalid-name
        sex_y_conf = 1-conf_sorted_frac_male
        sex_y_ds = 1-ds_sorted_frac_male
        sex_baseline = 1 - (conf_num_males / IMGS_THIS_CLASS)

    elif mode == "young":
        minority_sex = "Male" # pylint: disable=invalid-name
        sex_y_conf = conf_sorted_frac_male
        sex_y_ds = ds_sorted_frac_male
        sex_baseline = conf_num_males / IMGS_THIS_CLASS


    # Plot sex results for class
    plt.plot(range(IMGS_THIS_CLASS), sex_y_conf, color='g', label="Confidence")
    plt.plot(range(IMGS_THIS_CLASS), sex_y_ds, color='b', label="Decision Score")
    plt.axhline(y=sex_baseline, color='r', label="Baseline")
    plt.ylabel(f'Fraction {minority_sex}')
    plt.xlabel("Top K Flagged")
    plt.legend(loc="upper right")
    plt.title(f"{minority_sex} Flagged for Class {mode}")
    plt.savefig(f'{mode}_orig_results.png')
