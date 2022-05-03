import shutil
import csv
import os
import torch
import re
from random import randint, seed


seed(26) # 26


src_path = r"C:\Users\Marc\Desktop\model_data\\"
dst_train_path = r"C:\Users\Marc\Desktop\model_data\train_model_data\\"
dst_val_path = r"C:\Users\Marc\Desktop\model_data\val_model_data\\"
dst_test_path = r"C:\Users\Marc\Desktop\model_data\test_model_data\\"

#src_path = r"/home/tyson/model_data/"
#dst_train_path = r"/home/tyson/train_model_data/"
#dst_val_path = r"/home/tyson/val_model_data/"
#dst_test_path = r"/home/tyson/test_model_data/"

patient_samples = dict()

files = [] # read the old encoding file so that new ones can be made


with open(src_path + '\data_encoding.csv', 'r') as file:
    ls = csv.reader(file)
    for rows in ls:
        files.append(rows)
        m = re.match(r".+/\d+/(\d+)/+.", rows[0])
        val_to_dict = patient_samples.get(m.group(1), [])
        val_to_dict.append(rows[2])
        patient_samples[m.group(1)] = val_to_dict


print("Number of patients:", len(patient_samples))

train_size = 195 # 145
val_size = 28 # 21
test_size = 56 # 41

train_set = []
val_set = []
test_set = []

data_list = []


nr_of_series = 0 # total number of experiments

many_series = 0
nr_series = 0
for i in patient_samples:
    if (w := len(patient_samples[i])) > 3:
        many_series += 1
        nr_series += w
        train_size -= w
        train_set.append(patient_samples[i])
    else:
        data_list.append(patient_samples[i])
    nr_of_series += w

print()
print("Amount of patients with more than 3 experiments:", many_series)
print("This is a total amount of series with:", nr_series)


print()
print("The grand total of experiments:", nr_of_series)


data_list = sorted(data_list, key = len, reverse = True)

while data_list:
    value = data_list[0]
    value_size = len(value)
    rand_nr = randint(0, 2)

    if rand_nr == 2:
        if train_size >= value_size:
            train_set.append(value)
            train_size -= value_size
            data_list.remove(value)
    elif rand_nr == 1:
        if val_size >= value_size:
            val_set.append(value)
            val_size -= value_size
            data_list.remove(value)
    else:
        if test_size >= value_size:
            test_set.append(value)
            test_size -= value_size
            data_list.remove(value)

print()

print("Number of patients in the train set:", len(train_set))
print("Number of patients in the validation set:", len(val_set))
print("Number of patients in the test set:", len(test_set))




# first move the test files
set = sorted([int(i) for i in sum(test_set, [])])


j = 1
for i in set:
    # move input:
    shutil.move(src_path + f"model_input ({i}).pt",
    dst_test_path + f"model_input ({j}).pt")

    # move target:
    shutil.move(src_path + f"model_target ({i}).pt",
    dst_test_path + f"model_target ({j}).pt")
    j += 1

# create a test encoding file
with open(dst_test_path + "/test_encoding.csv", "w", newline='') as f:
    write = csv.writer(f)
    j = 1
    for i in set:
        write.writerow([files[i - 1][0], files[i - 1][1], j])
        j += 1



set = sorted([int(i) for i in sum(train_set, [])])

j = 1
for i in set:
    # rename input:
    os.rename(src_path + f"model_input ({i}).pt",
    dst_train_path + f"model_input ({j}).pt")

    # rename target:
    os.rename(src_path + f"model_target ({i}).pt",
    dst_train_path + f"model_target ({j}).pt")
    j += 1

# create a train encoding file
with open(dst_train_path + "/train_encoding.csv", "w", newline='') as f:
    write = csv.writer(f)
    j = 1
    for i in set:
        write.writerow([files[i - 1][0], files[i - 1][1], j])
        j += 1



set = sorted([int(i) for i in sum(val_set, [])])

j = 1
for i in set:
    # rename input:
    os.rename(src_path + f"model_input ({i}).pt",
    dst_val_path + f"model_input ({j}).pt")

    # rename target:
    os.rename(src_path + f"model_target ({i}).pt",
    dst_val_path + f"model_target ({j}).pt")
    j += 1

# create a train encoding file
with open(dst_val_path + "/val_encoding.csv", "w", newline='') as f:
    write = csv.writer(f)
    j = 1
    for i in set:
        write.writerow([files[i - 1][0], files[i - 1][1], j])
        j += 1
