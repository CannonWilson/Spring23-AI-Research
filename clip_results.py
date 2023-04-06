from age_model import CustomAgeNetwork
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import clip
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import svm
import csv

print('Initializing Models')

# Custom model
CUSTOM_MODEL_PATH = './my_age_model.pth'
custom_model = CustomAgeNetwork()
custom_model.load_state_dict(torch.load(CUSTOM_MODEL_PATH))

data_transforms = transforms.Compose([
    transforms.Resize((82, 100)),
    transforms.ToTensor()
])

BASE_DIR = './val_new' # SVM is trained on *validation* set
TRAIN_DIR = './train_new'
val_data = datasets.ImageFolder(BASE_DIR, transform=data_transforms)
val_loader = DataLoader(val_data, batch_size=1)
NUM_CLASSES = 2
TOTAL_TRAIN_IMAGES = len(DataLoader(datasets.ImageFolder(TRAIN_DIR), batch_size=1))

total = 0
for root, dirs, files in os.walk(BASE_DIR):
    total += len(files)
TOTAL_TRAIN_IMAGES = total


# CLIP
EMBEDDING_DIM = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Other variables
img_feature_stack = torch.empty(TOTAL_TRAIN_IMAGES, EMBEDDING_DIM).to(device)
correctness = torch.empty(TOTAL_TRAIN_IMAGES, dtype=torch.int8).to(device)
file_paths = []

# Get the feature embeddings and calculate correctness 
# for every training image
with torch.no_grad():
    custom_model.eval()
    clip_model.eval()
    print('Calculating CLIP embeddings and correctness for training images')
    for i, (image, label) in enumerate(val_loader, 0):

        # Calculate correctness of custom model
        model_output = custom_model(image)
        _, predicted = torch.max(model_output.data, 1)
        pred = predicted.item()
        actual = label.item()
        if pred == actual:
            correctness[i] = 1
        else: 
            correctness[i] = -1

        # Store file path
        path = val_loader.dataset.samples[i][0]
        file_paths.append(path)

        # Generate CLIP embedding
        pil_image = Image.open(path)
        clip_img = clip_preprocess(pil_image).unsqueeze(0).to(device)
        img_features = clip_model.encode_image(clip_img)
        img_feature_stack[i] = img_features

print('Finished getting clip embeddings and correctness scores.')
print('Beginning to fit SVM classifiers')

SHOW_RESULTS = False
SAVE_RESULTS = True
RESULTS_FILE = './4-4_results.csv'
last_seen_class = file_paths[0].split('/')[2] # either 'old' or 'young'
start_idx = 0
end_idx = 0

if SAVE_RESULTS: # Setup for save
    f = open(RESULTS_FILE, 'w')
    writer = csv.writer(f)
    header_str = 'filename,5_o_Clock_Shadow,Arched_Eyebrows,Attractive,Bags_Under_Eyes,Bald,Bangs,Big_Lips,Big_Nose,Black_Hair,Blond_Hair,Blurry,Brown_Hair,Bushy_Eyebrows,Chubby,Double_Chin,Eyeglasses,Goatee,Gray_Hair,Heavy_Makeup,High_Cheekbones,Male,Mouth_Slightly_Open,Mustache,Narrow_Eyes,No_Beard,Oval_Face,Pale_Skin,Pointy_Nose,Receding_Hairline,Rosy_Cheeks,Sideburns,Smiling,Straight_Hair,Wavy_Hair,Wearing_Earrings,Wearing_Hat,Wearing_Lipstick,Wearing_Necklace,Wearing_Necktie,Young'
    writer.writerow(['DS_Score', 'Correctness', *header_str.split(',')])
    celeba_df = pd.read_csv('list_attr_celeba.csv')

for i, img_path in enumerate(file_paths):
    f_name = img_path.split('/')[-1] # change to os.path.split
    cur_class = img_path.split('/')[2]
    end_idx += 1
    if (cur_class != last_seen_class) or (i == len(file_paths) - 1):
        print('Fitting SVM for class:', last_seen_class)

        # Fit the SVM on all of the embeddings
        svm_classifier = svm.SVC(kernel="linear")
        svm_classifier.fit(img_feature_stack[start_idx:end_idx], \
            correctness[start_idx:end_idx])
        
        # Then, loop over the CLIP embeddings for the class and 
        # store the decision scores
        print('Calculating decision scores for class: ', last_seen_class)
        num_images_in_class = end_idx - start_idx
        decision_scores = np.empty(num_images_in_class)
        for idx in range(num_images_in_class):
            score = np.dot(svm_classifier.coef_[0].transpose(), \
                img_feature_stack[idx].numpy()) + \
                    svm_classifier.intercept_
            decision_scores[idx] = score[0]

        # Sort the decision scores and show a graphic illustrating
        # a sample of the removed images
        if SHOW_RESULTS:
            print('Showing results: ')
            NUM_SAMPLES = 20 # number of samples to show per class
            sorted_idxs = np.argsort(decision_scores)
            spaced_idxs = np.int32(np.linspace(0, len(sorted_idxs)-1, NUM_SAMPLES))
            selected_idxs = start_idx + sorted_idxs[spaced_idxs]

            fig = plt.figure(figsize=(1, NUM_SAMPLES)) # Show everything in one run
            fig.suptitle(f'Results for class: {last_seen_class}', fontsize=16)
            for col_idx, idx in enumerate(selected_idxs):
                fig.add_subplot(1, NUM_SAMPLES, col_idx+1)
                plt.imshow(Image.open(file_paths[idx]))
                plt.axis('off')
                title_msg = f"DS: {round(1000*decision_scores[idx - start_idx])/1000}\n" + \
                    f"Co: {correctness[idx]}"  
                plt.title(title_msg, fontdict={'family':'serif','color':'black','size':8})

        # Save decision scores and attributes to csv file
        if SAVE_RESULTS:
            print('Saving results to csv file')
            
            for idx, f_path in enumerate(file_paths[start_idx:end_idx]):
                f_name = f_path.split('/')[-1]
                f_row = celeba_df[celeba_df['filename'] == f_name]
                attr_vals = list(f_row.iloc[0])
                row = [decision_scores[idx].item(), correctness[idx].item(), *attr_vals]
                writer.writerow(row)

        # Increment variables for next class
        last_seen_class = cur_class
        start_idx = end_idx

if SAVE_RESULTS:
    f.close()

plt.show()