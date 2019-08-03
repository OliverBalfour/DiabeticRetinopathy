
from dataframes import get_train_df_old, get_train_df_new, get_train_df
from augment_utils import map_df, get_augment_seq, generate_augmentations, process_test_set, remove_used

"""
We need to generate 3 datasets:
 - severity/train - 10K total, 1K per 5 classes per 2 datasets, including augmentations.
 - binary/train   - 20K total, 5K per 2 class groups (0 or 1,2,3,4) per 2 datasets, including augmentations if needed (unequal weight for 1,2,3,4).
 - binary/test    - 2K total, 1K per class, no fixed quota per dataset. NO OVERLAP OR AUGMENTATIONS ALLOWED.

Each dataset is in folders like data/proc/severity/train/224/3 and has subfolders either 224/ 0,1,2,3,4 or 224/ 0,1

The map_df function has a parameter to collapse classes into one for binary datasets.
To ensure no test set leakage, binary/test is generated first and the files it used are removed from all of the dataframes before being used for the other datasets.
It does not matter if severity/train and binary/train overlap at all.
"""

seq = get_augment_seq()
new_df = get_train_df_new()
old_df = get_train_df_old()
all_df = get_train_df()

### binary/test

used = process_test_set(map_df(all_df, 'data/proc/binary/test/', binary=True), seq, class_size=1000)
new_df = remove_used(new_df, used)
old_df = remove_used(old_df, used)
all_df = remove_used(all_df, used)
print('\nSelected images for binary/test and removed them from the dataframe to avoid test set leakage.')

### binary/train

generate_augmentations(map_df(new_df, 'data/proc/binary/train/', binary=True), seq, class_size=5000)
print('\nAugmented new training images in binary/train.')

generate_augmentations(map_df(old_df, 'data/proc/binary/train/', binary=True), seq, class_size=5000)
print('\nAugmented old training images in binary/train.')

### severity/train

generate_augmentations(map_df(new_df, 'data/proc/severity/train/'), seq, class_size=1000)
print('\nAugmented new training images in severity/train.')

generate_augmentations(map_df(old_df, 'data/proc/severity/train/'), seq, class_size=1000)
print('\nAugmented old training images in severity/train.')
