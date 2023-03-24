import os
import shutil

# set the directories where the images and label files are stored
image_dir = "/Users/monkey/Public/Python/New_Dataset/test_big_data/val/images"
label_dir = "/Users/monkey/Public/Python/New_Dataset/labels_part2"

# set the directory where the matching label files will be copied to
copy_dir = "/Users/monkey/Public/Python/New_Dataset/test_big_data/val/labels"

# iterate over the files in the image directory
for filename in os.listdir(image_dir):
    # split the file name and extension
    name, ext = os.path.splitext(filename)

    # search for files with the same name but any extension in the label directory
    label_filenames = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.startswith(name)]

    if label_filenames:
        # if there is at least one matching file, copy the first one to the copy directory
        shutil.copy2(label_filenames[0], copy_dir)
    else:
        # if there is no matching file, print the file name
        print("No label file found for: {}".format(filename))
