from os import listdir
from os.path import isfile, join

import torch
import os
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
import collections

from sklearn.model_selection import train_test_split
from config import args
from utils.preprocessing import preprocess_sample


def set_device():
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    return device


# def remove_lrs3_nuggets(filesList):
#     print("Removing `{LG}` and such")
#     for file in filesList:
#         txt = file + ".txt"
#         print("File")
#         current_example = open(txt, 'r')
#         lines = file1.readlines()
#         for line in lines:
#             if "{" in line:
#                 line = line.split("{")[0]
#         with open("test.txt", "w") as f:
#             f.writelines(lines)


def string_filter(file, folder, val=False):
    if val:
        dir = args["TEST_DIRECTORY"] + folder + "/" + file
    else:
        dir = args["TRAIN_DIRECTORY"] + folder + "/" + file
    with open(dir, "r") as f:
        lines = f.readlines()
        string_to_add = str(lines[0][6: -1])
        if "{" in string_to_add:
            string_to_add = lrs3_parse(string_to_add)
        characters = len([ele for ele in string_to_add if ele.isalpha()])
        print("String :" + str(string_to_add))
        print("characters :" + str(characters) + " and MAX is " + str(args["MAX_CHAR_LEN"]))
            # print(characters,args["MAX_CHAR_LEN"])
        if characters <= args["MAX_CHAR_LEN"]:
            return True
    return False


def check_valid_dirs(fileList):
    valid_dirs = list()
    for file in fileList:
        if args["TRAIN_DIRECTORY"] in file:
            print(file)
            print(file)
            print(os.path.isdir(file[:-5]))
            txt = file + ".txt"
            if (os.path.isdir(file[:-5]) and os.path.isfile(txt)):
                valid_dirs.append(file)
        # os.path.isdir()
    return valid_dirs


def filer_lengths(fileList):
    filesListFiltered = list()
    for file in fileList:
        text = file + ".txt"
        with open(text, "r") as f:
            lines = f.readlines()
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
                characters = len([ele for ele in string_to_add if ele.isalpha()])
                if characters <= args["MAX_CHAR_LEN"]:
                    filesListFiltered.append(file)

    print("\nNumber of data samples to after filtering = %d" % (len(filesListFiltered)))
    return filesListFiltered


def get_filelist():
    # walking through the data directory and obtaining a list of all files in the dataset
    filesList = list()
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
        # print(root,dirs,files)
        for file in files:
            if file.endswith(".mp4"):
                # if check_len(file,args["MAX_CHAR_LEN"]):
                filesList.append(os.path.join(root, file[:-4]))
    # print(filesList)
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")
    return filesList


def generate_noise_file(filesList):
    # Generating a 1 hour noise file
    # Fetching audio samples from 20 random files in the dataset and adding them up to generate noise
    # The length of these clips is the shortest audio sample among the 20 samples
    print("\n\nGenerating the noise file ....")

    noise = np.empty((0))
    while len(noise) < 16000 * 3600:
        noisePart = np.zeros(16000 * 60)
        indices = np.random.randint(0, len(filesList), 20)
        for ix in indices:
            sampFreq, audio = wavfile.read(filesList[ix] + ".wav")
            audio = audio / np.max(np.abs(audio))
            pos = np.random.randint(0, abs(len(audio) - len(noisePart)) + 1)
            if len(audio) > len(noisePart):
                noisePart = noisePart + audio[pos:pos + len(noisePart)]
            else:
                noisePart = noisePart[pos:pos + len(audio)] + audio
        noise = np.concatenate([noise, noisePart], axis=0)
    noise = noise[:16000 * 3600]
    noise = (noise / 20) * 32767
    noise = np.floor(noise).astype(np.int16)
    wavfile.write(args["DATA_DIRECTORY"] + "/noise.wav", 16000, noise)

    print("\nNoise file generated.")


def preprocess_all_samples(filesList):
    # declaring the visual frontend module
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file)
    print("\nPreprocessing Done.")


def lrs3_parse(example):
    splt = example.split("{")
    print("Had to split")
    print("Was " + str(example))
    print("Is "+ str(splt))
    return splt[0]

def generate_train_file(train):
    # Generating train.txt for splitting the pretrain set into train sets
    train_dir_file = args["DATA_DIRECTORY"] + "/train.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("Generating train file...")
    for train_dir in tqdm(train):
        text_file = train_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            example_dict["ID"].append(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            example_dict["TEXT"].append(string_to_add)
            # print(example_dict)

    if os.path.isfile(train_dir_file):
        os.remove(train_dir_file)
    with open(train_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")

def generate_val_file(val):
    # Generating val.txt for splitting the pretrain set into validation sets
    val_dir_file = args["DATA_DIRECTORY"] + "/val.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("Generating val file...")
    for val_dir in tqdm(val):
        text_file = val_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            example_dict["ID"].append(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            example_dict["TEXT"].append(string_to_add)
            # print(example_dict)

    if os.path.isfile(val_dir_file):
        os.remove(val_dir_file)

    with open(val_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")

def generate_test_file(test):
    # Generating train.txt for splitting the pretrain set into train sets
    test_dir_file = args["DATA_DIRECTORY"] + "/test.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("Generating test file...")
    for test_dir in tqdm(test):
        text_file = test_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            example_dict["ID"].append(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            example_dict["TEXT"].append(string_to_add)
            # print(example_dict)

    if os.path.isfile(test_dir_file):
        os.remove(test_dir_file)
    with open(test_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")


def generate_extended_test_file(test,pretrain):
    # Generating val.txt for splitting the pretrain set into validation sets
    test_dir = args["TEST_DIRECTORY"]
    pretrain_dir = args["PRETRAIN_AUDIO_MODEL"]
    test_dir_file = args["DATA_DIRECTORY"] + "/extended_test.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("Generating extrended test file...")
    for test_dir in tqdm(test):
        text_file = test_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            if len(string_to_add) < args["MAX_CHAR_LEN"]:
                example_dict["ID"].append(examples_npy_dir)
                example_dict["TEXT"].append(string_to_add)
            # print(example_dict)

    for test_dir in tqdm(pretrain):
        text_file = test_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            if len(string_to_add) < args["MAX_CHAR_LEN"]:
                example_dict["ID"].append(examples_npy_dir)
                example_dict["TEXT"].append(string_to_add)

            # print(example_dict)

    if os.path.isfile(test_dir_file):
        os.remove(test_dir_file)
    with open(test_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")

def generate_extended_train_file(train,pretrain):
    # Generating val.txt for splitting the pretrain set into validation sets
    train_dir_file = args["DATA_DIRECTORY"] + "/extended_train.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    train_NOT_IN_USE = 0
    pretrain_NOT_IN_USE = 0
    print("Generating extrended train file...")
    for train_dir in tqdm(train):
        text_file = train_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            examples_npy_dir = text_file.split("txt")[0][:-1]
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            if args["MAX_CHAR_LEN"] > len(string_to_add) >= 2:
                example_dict["ID"].append(examples_npy_dir)
                example_dict["TEXT"].append(string_to_add)
            else:
                train_NOT_IN_USE +=1

    for test_dir in tqdm(pretrain):
        text_file = test_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            examples_npy_dir = text_file.split("txt")[0][:-1]
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            if args["MAX_CHAR_LEN"] > len(string_to_add) >= 2:
                example_dict["ID"].append(examples_npy_dir)
                example_dict["TEXT"].append(string_to_add)
            else:
                pretrain_NOT_IN_USE +=1

    if os.path.isfile(train_dir_file):
        os.remove(train_dir_file)
    with open(train_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")

    return train_NOT_IN_USE , pretrain_NOT_IN_USE


def split_trainval(fileList):
    trainval_only = [x for x in fileList if (args["TRAIN_SET_NAME"] in x)]
    print("We have a total of :" + str(len(trainval_only)))
    print("Now we want a split 80/20")
    train, val = train_test_split(trainval_only, test_size=.20, shuffle=False)
    if collections.Counter(train) == collections.Counter(val):
        print("WARNING: TRAIN AND VAL HAVE SOME OVERLAPPING ELEMENTS")
        exit()
    return train, val

def check_files_correct_len(train, val, test,pretrain,extended_train_NOT_IN_USE):
    train_len = len(train)
    val_len = len(val)
    test_len = len(test)
    extended_test_len = len(test) + len(pretrain)
    extended_train_len = len(train) + len(pretrain)

    extended_train_len_NOT_IN_USE = extended_train_NOT_IN_USE[0] + extended_train_NOT_IN_USE[1]

    val_dir_file = args["DATA_DIRECTORY"] + "/val.txt"
    train_dir_file = args["DATA_DIRECTORY"] + "/train.txt"
    test_dir_file = args["DATA_DIRECTORY"] + "/test.txt"
    extnended_test_dir_file = args["DATA_DIRECTORY"] + "/extended_test.txt"
    extnended_train_dir_file = args["DATA_DIRECTORY"] + "/extended_train.txt"

    with open(train_dir_file) as f:
        text = f.readlines()
        train_file_len = len(text)

    with open(val_dir_file) as f:
        text = f.readlines()
        val_file_len = len(text)

    with open(test_dir_file) as f:
        text = f.readlines()
        test_file_len = len(text)

    with open(extnended_test_dir_file) as f:
        text = f.readlines()
        extended_test_file_len = len(text)

    with open(extnended_train_dir_file) as f:
        text = f.readlines()
        extended_train_file_len = len(text)

    print("Expected train len: " + str(train_len) + " Got train len: " + str(train_file_len))
    print("Expected val len: " + str(val_len) + " Got val len: " + str(val_file_len))
    print("Expected test len: " + str(test_len) + " Got test len: " + str(test_file_len))
    print("Should be lower due to string chopping to max length")
    print("Expected extended test len: " + str(extended_test_len) + " Got test len: " + str(extended_test_file_len))
    print("Expected extended train len: " + str(extended_train_len - extended_train_len_NOT_IN_USE) + " Got test len: " + str(extended_train_file_len))

def remove_null_examples(file):
    with open(file) as f:
        for line in f:
            print(line)



if __name__ == "__main__":
    device = set_device()
    fileList = get_filelist()
    print("File List complete")
    train, val = split_trainval(fileList)
    test = [x for x in fileList if (args["TEST_SET_NAME"] in x)]
    pretrain = [x for x in fileList if (args["PRETRAIN_SET_NAME"] in x)]
    pretrain = pretrain[:args["PRETRAIN_SET_SIZE"] ]
    print("Size of train set" + str(len(train)))
    print("Size of val set:" + str(len(val)))
    print("Size of test set:" + str(len(test)))
    # preprocess_all_samples(fileList)
    # generate_noise_file(fileList)
    # preprocess_all_samples(fileList,device)
    # generate_train_file(train)
    # generate_val_file(val)
    # generate_test_file(test)
    generate_extended_test_file(test,pretrain)
    train_NOT_IN_USE , pretrain_NOT_IN_USE = generate_extended_train_file(train,pretrain)
    check_files_correct_len(train, val, test, pretrain,[train_NOT_IN_USE,pretrain_NOT_IN_USE])
    print("Completed")
