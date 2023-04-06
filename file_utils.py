"""
This file contains some helper functions
that take in a file path that looks like this:
1_1_1_1_-1_-1_-1_-1_1_-1_-1_-1_-1_-1_-1_-1_-1_-1_-1_-1_1_-1_-1_-1_-1_-1_-1_1_-1_-1_1_-1_-1_-1_-1_-1_-1_-1_-1_-1_163507.jpg
and returns useful information about 
the attributes of this file.
"""

header = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"] 

def young_or_old(f_name):
    attrs = f_name.split('_')
    y_o_header_idx = header.index("Young")
    if attrs[y_o_header_idx] == "1":
        return 'young'
    else:
        return 'old'
    
def get_attr(f_name, col_name):
    attrs = f_name.split('_')
    header_idx = header.index(col_name)
    if attrs[header_idx] == "1":
        return True
    else:
        return False