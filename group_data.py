import os
import shutil

with open("list_attr_celeba.txt") as f:
    file_lines = f.read().split("\n")[2:]# first two lines are not data
    
    for line in file_lines[1:]:
        destination_dir = ""
        line_text = [x.strip() for x in line.split(' ') if x != '']
        filename = line_text[0]
        male_score = line_text[21] 
        young_score = line_text[-1] 

        if male_score == "1" and young_score == "1":
            destination_dir = "./young_male"
        elif male_score == "-1" and young_score == "1":
            destination_dir = "./young_female"
        elif male_score == "1" and young_score == "-1":
            destination_dir = "./old_male"
        elif male_score == "-1" and young_score == "-1":
            destination_dir = "./old_female"
        else: 
            continue

        if os.path.exists(os.path.join('img_align_celeba', filename)):
            if not os.path.exists(os.path.join(destination_dir, filename)):
                shutil.move(os.path.join('./img_align_celeba', filename), \
                    os.path.join(destination_dir, filename))

"""
After running this script, you should have
14878 old females
30987 old males
103287 young females
53447 young males

Next, following Saachi et al. (https://arxiv.org/pdf/2206.14754.pdf)
the split data should then be moved into training folders for
`6203 each of “old” “female” and “young” “male”, 
and 24812 of “old” “male” and “young” “female”`
a validation set with `1795 examples for each of the 
demographic categories` 
"""