import os

directory = "/Users/monkey/Public/Python/Science Research 2022- Bone Fracture Detection.v9i.voc/test_n" # Replace with the path to your directory
extensions_to_keep = [".jpg", ".png"]

for filename in os.listdir(directory):
    if filename.endswith(tuple(extensions_to_keep)):
        continue
    else:
        os.remove(os.path.join(directory, filename))
