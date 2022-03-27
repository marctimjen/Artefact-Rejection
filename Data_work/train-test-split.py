import shutil
import csv
import os
import torch


src_path = r"C:\Users\Marc\Desktop\model_data\\"
dst_path = r"C:\Users\Marc\Desktop\test_model_data\\"

#src_path = r"/home/tyson/model_data/"
#dst_path = r"/home/tyson/test_model_data/"

files = [] # read the old encoding file so that new ones can be made
with open(src_path + 'data_encoding.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        files.append(row)

# randomly split the data
test_set, train_set = torch.utils.data.random_split(
                                range(1, 284), [57, 226],
                                generator=torch.Generator().manual_seed(42))

# first move the test files
set = sorted([i for i in test_set])


j = 1
for i in set:
    # move input:
    shutil.move(src_path + f"model_input ({i}).pt",
    dst_path + f"model_input ({j}).pt")

    # move target:
    shutil.move(src_path + f"model_target ({i}).pt",
    dst_path + f"model_target ({j}).pt")
    j += 1

# create a test encoding file
with open(dst_path + "/test_encoding.csv", "w", newline='') as f:
    write = csv.writer(f)
    j = 1
    for i in set:
        write.writerow([files[i - 1][0], files[i - 1][1], j])
        j += 1

set = sorted([i for i in train_set])

j = 1
for i in set:
    # rename input:
    os.rename(src_path + f"model_input ({i}).pt",
    src_path + f"model_input ({j}).pt")

    # rename target:
    os.rename(src_path + f"model_target ({i}).pt",
    src_path + f"model_target ({j}).pt")
    j += 1

# create a train encoding file
with open(src_path + "/train_encoding.csv", "w", newline='') as f:
    write = csv.writer(f)
    j = 1
    for i in set:
        write.writerow([files[i - 1][0], files[i - 1][1], j])
        j += 1
