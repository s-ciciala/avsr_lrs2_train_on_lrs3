"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import numpy as np
from tqdm import tqdm

from .metrics import compute_cer, compute_wer
from .decoders import ctc_greedy_decode, ctc_search_decode
from config import args


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function, device, trainParams):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    trainingCER = 0
    trainingWER = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                          ncols=75)):

        inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        opmode = np.random.choice(["AO", "VO", "AV"],
                                  p=[trainParams["aoProb"], trainParams["voProb"], 1-(trainParams["aoProb"]+trainParams["voProb"])])
        if opmode == "AO":
            inputBatch = (inputBatch[0], None)
        elif opmode == "VO":
            inputBatch = (None, inputBatch[1])
        else:
            pass

        optimizer.zero_grad()
        model.train()
        outputBatch = model(inputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            arry = []
            for btch in inputLenBatch:
                # print("HERE")
                # print(btch)
                if len(outputBatch) < btch:
                    arry.append(len(outputBatch))
                else:
                    arry.append(btch)
            new_inputLenBatch = torch.tensor(arry, dtype=torch.int32, device=device)
            loss = loss_function(outputBatch, targetBatch, new_inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch, trainParams["eosIx"])
        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, trainParams["spaceIx"])

    trainingLoss = trainingLoss/len(trainLoader)
    trainingCER = trainingCER/len(trainLoader)
    trainingWER = trainingWER/len(trainLoader)
    return trainingLoss, trainingCER, trainingWER



# def evaluate(model, evalLoader, loss_function, device, evalParams):
#
#     """
#     Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
#     The CTC decode scheme can be set to either 'greedy' or 'search'.
#     """
#
#     evalLoss = 0
#     evalCER = 0
#     evalWER = 0
#     predictionStrings = []
#     targetStrings = []
    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
                                                                                          ncols=75)):

        inputBatch, targetBatch = ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        opmode = np.random.choice(["AO", "VO", "AV"],
                                  p=[evalParams["aoProb"], evalParams["voProb"], 1-(evalParams["aoProb"]+evalParams["voProb"])])
        if opmode == "AO":
            inputBatch = (inputBatch[0], None)
        elif opmode == "VO":
            inputBatch = (None, inputBatch[1])
        else:
            pass

        model.eval()
        with torch.no_grad():
            outputBatch = model(inputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                arry = []
                for btch in inputLenBatch:
                    # print("HERE")
                    # print(btch)
                    if len(outputBatch) < btch:
                        arry.append(len(outputBatch))
                    else:
                        arry.append(btch)
                new_inputLenBatch = torch.tensor(arry, dtype=torch.int32, device=device)
                loss = loss_function(outputBatch, targetBatch, new_inputLenBatch, targetLenBatch)

        evalLoss = evalLoss + loss.item()
        if evalParams["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, evalParams["eosIx"])
        elif evalParams["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch, evalParams["beamSearchParams"],
                                                                    evalParams["spaceIx"], evalParams["eosIx"], evalParams["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, evalParams["spaceIx"])

        index_to_char = args["INDEX_TO_CHAR"]
        predictionString = ""
        for i in range(len(predictionBatch)):
            item_idx = predictionBatch[i].item()
            charrr = index_to_char[item_idx]
            predictionString += str(charrr)
        predictionStrings.append(predictionString)
        targetString = ""
        for i in range(len(targetBatch)):
            item_idx = targetBatch[i].item()
            charrr = index_to_char[item_idx]
            targetString += str(charrr)
        targetStrings.append(targetString)

    evalLoss = evalLoss/len(evalLoader)
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    if args["DISPLAY_PREDICTIONS"]:
        for i in range(len(predictionStrings)):
            print("------------------PREDICTION------------------")
            print("------------------PREDICTION------------------")
            print(predictionStrings[i])
            print("------------------TARGET------------------")
            print("------------------TARGET------------------")
            print(targetStrings[i])
        with open(args["AUDIO_VISUAL_RESULTS_TEXT_FILENAME"] , 'w') as f:
            for i in range(len(predictionStrings)):
                f.write("------------------TARGET------------------\n")
                f.write("%s\n" % str(targetStrings[i]))
                f.write("------------------PREDICTION------------------\n")
                f.write("%s\n" % str(predictionStrings[i]))
            f.write("\n" + "evalLoss: " + str(evalLoss))
            f.write("\n" + "evalCER: " + str(evalCER))
            f.write("\n" + "evalWER: " + str(evalWER))

    return evalLoss, evalCER, evalWER
