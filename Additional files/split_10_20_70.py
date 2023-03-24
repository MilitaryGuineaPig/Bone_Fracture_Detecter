import os
import random
import shutil


def split_dataset(dir_path):
    test_dir = os.path.join(dir_path, 'test')
    val_dir = os.path.join(dir_path, 'val')
    train_dir = os.path.join(dir_path, 'train')

    # Create directories if they don't exist
    for dir in [test_dir, val_dir, train_dir]:
        os.makedirs(os.path.join(dir, 'images'), exist_ok=True)

    for filename in os.listdir(dir_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            src = os.path.join(dir_path, filename)
            if random.random() < 0.1:
                dst = os.path.join(test_dir, 'images', filename)
                shutil.copyfile(src, dst)
            elif random.random() < 0.2:
                dst = os.path.join(val_dir, 'images', filename)
                shutil.copyfile(src, dst)
            else:
                dst = os.path.join(train_dir, 'images', filename)
                shutil.copyfile(src, dst)


dirPath = "/Users/monkey/Public/Python/New_Dataset/images"
# example usage
split_dataset(dirPath)
