from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
from age_model import CustomAgeNetwork

print('Initializing Models')

# Custom model
CUSTOM_MODEL_PATH = './smiling_age_model.pth'
custom_model = CustomAgeNetwork()
custom_model.load_state_dict(torch.load(CUSTOM_MODEL_PATH))

data_transforms = transforms.Compose([
    transforms.Resize((82, 100)),
    transforms.ToTensor()
])

# Top_K is evaluated on *test* set
TEST_DIR = './test'
test_data = datasets.ImageFolder(TEST_DIR, transform=data_transforms)
test_loader = DataLoader(test_data, batch_size=1)
NUM_CLASSES = 2
MODES = ["old", "young"]

# CLIP
EMBEDDING_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

            # Make sure image belongs to current class
            # being considered (young/old). Skip for now if not
            path = test_loader.dataset.samples[i][0] # ex: ./val/old/female/no_smile/000997.jpg
            img_class = path.split('/')[2]
            if not mode == img_class:
                continue

            # Calculate correctness/confidence of custom model
            model_output = custom_model(image)
            confidence, predicted = torch.max(model_output.data, 1)
            conf = confidence.item()
            confidences.append(conf) # [cur_idx] = conf
            pred = predicted.item()
            actual = label.item()
            if pred == actual:
                # correctness[cur_idx] = 1
                correctness.append(1)
            else:
                # correctness[cur_idx] = -1
                correctness.append(-1)

            # Record sex/smiling of current img
            sex = path.split('/')[-3]
            smiling = path.split('/')[-2]
            sexes.append(1 if sex == "male" else 0)
            smiles.append(1 if smiling == "smile" else 0)

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
    svm_classifier.fit(torch.cat(img_feature_stack), torch.tensor(correctness, dtype=torch.int8))
    for idx in range(IMGS_THIS_CLASS):
        score = np.dot(svm_classifier.coef_[0], \
            img_feature_stack[idx].numpy().transpose()) + \
                svm_classifier.intercept_
        ds_values.append(score[0])

    print('Fitted SVM, now plotting results.')
    conf_sorted_idxs =  np.argsort(confidences)
    ds_sorted_idxs = np.argsort(ds_values)
    conf_sorted_sexes = np.array(sexes)[conf_sorted_idxs]
    ds_sorted_sexes = np.array(sexes)[ds_sorted_idxs]
    conf_sorted_smiles = np.array(smiles)[conf_sorted_idxs]
    ds_sorted_smiles = np.array(smiles)[ds_sorted_idxs]
    conf_sorted_frac_male = np.empty(IMGS_THIS_CLASS)
    ds_sorted_frac_male = np.empty(IMGS_THIS_CLASS)
    conf_sorted_frac_smiles = np.empty(IMGS_THIS_CLASS)
    ds_sorted_frac_smiles = np.empty(IMGS_THIS_CLASS)

    for num_people in range(IMGS_THIS_CLASS):
        conf_num_males = conf_sorted_sexes[:num_people].sum()
        ds_num_males = ds_sorted_sexes[:num_people].sum()
        conf_num_smiles = conf_sorted_smiles[:num_people].sum()
        ds_num_smiles = ds_sorted_smiles[:num_people].sum()
        conf_sorted_frac_male[num_people] = conf_num_males / num_people
        ds_sorted_frac_male[num_people] = ds_num_males / num_people
        conf_sorted_frac_smiles[num_people] = conf_num_smiles / num_people
        ds_sorted_frac_smiles[num_people] = ds_num_smiles / num_people

    # make sure that the final tally for the number of males is the same
    assert conf_num_males == ds_num_males 
    assert conf_num_smiles == ds_num_smiles

    if mode == "old":
        minority_sex = "Female" # pylint: disable=invalid-name
        sex_y_conf = 1-conf_sorted_frac_male
        sex_y_ds = 1-ds_sorted_frac_male
        sex_baseline = 1 - (conf_num_males / IMGS_THIS_CLASS)

        minority_smile = "Smiling" # pylint: disable=invalid-name
        smi_y_conf = conf_sorted_frac_smiles
        smi_y_ds = ds_sorted_frac_smiles
        smi_baseline = conf_num_smiles / IMGS_THIS_CLASS

    elif mode == "young":
        minority_sex = "Male" # pylint: disable=invalid-name
        sex_y_conf = conf_sorted_frac_male
        sex_y_ds = ds_sorted_frac_male
        sex_baseline = conf_num_males / IMGS_THIS_CLASS

        minority_smile = "Not Smiling" # pylint: disable=invalid-name
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
    plt.show()

    # Plot smiling results for class
    plt.plot(range(IMGS_THIS_CLASS), smi_y_conf, color='g', label="Confidence")
    plt.plot(range(IMGS_THIS_CLASS), smi_y_ds, color='b', label="Decision Score")
    plt.axhline(y=smi_baseline, color='r', label="Baseline")
    plt.ylabel(f'Fraction {minority_smile}')
    plt.xlabel("Top K Flagged")
    plt.legend(loc="upper right")
    plt.title(f"{minority_smile} Flagged for Class {mode}")
    plt.show()
