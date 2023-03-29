"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
import numpy as np
import editdistance
from config import args


def compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch):

    """
    Function to compute the Character Error Rate using the Predicted character indices and the Target character
    indices over a batch.
    CER is computed by dividing the total number of character edits (computed using the editdistance package)
    with the total number of characters (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the CER.
    """

    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
    totalEdits = 0
    totalChars = 0

    for n in range(len(preds)):
        pred = preds[n].numpy()[:-1]
        trgt = trgts[n].numpy()[:-1]
        numEdits = editdistance.eval(pred, trgt)
        totalEdits = totalEdits + numEdits
        totalChars = totalChars + len(trgt)

    return totalEdits/totalChars



# def compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):
#
#     """
#     Function to compute the Word Error Rate using the Predicted character indices and the Target character
#     indices over a batch. The words are obtained by splitting the output at spaces.
#     WER is computed by dividing the total number of word edits (computed using the editdistance package)
#     with the total number of words (total => over all the samples in a batch).
#     The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
#     """
#
#     targetBatch = targetBatch.cpu()
#     targetLenBatch = targetLenBatch.cpu()
#
#     preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
#     trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
#     totalEdits = 0
#     totalWords = 0
#
#     for n in range(len(preds)):
#         pred = preds[n].numpy()[:-1]
#         trgt = trgts[n].numpy()[:-1]
#
#         predWords = np.split(pred, np.where(pred == spaceIx)[0])
#         predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]
#
#         trgtWords = np.split(trgt, np.where(trgt == spaceIx)[0])
#         trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]
#
#         numEdits = editdistance.eval(predWords, trgtWords)
#         totalEdits = totalEdits + numEdits
#         totalWords = totalWords + len(trgtWords)
#
#     return totalEdits/totalWords
def compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):

    """
    Function to compute the Word Error Rate using the Predicted character indices and the Target character
    indices over a batch. The words are obtained by splitting the output at spaces.
    WER is computed by dividing the total number of word edits (computed using the editdistance package)
    with the total number of words (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
    """

    targetBatch = targetBatch.cpu()
    targetLenBatch = targetLenBatch.cpu()
    # print("Walking through and example...")

    preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
    trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))

    # print("Predictions " + str(preds))
    # print("Targets " + str(trgts))
    totalEdits = 0
    totalWords = 0
    index_to_char = args["INDEX_TO_CHAR"]

    for n in range(len(preds)):
        pred = preds[n].numpy()[:-1]
        trgt = trgts[n].numpy()[:-1]

        #TURN TO INTS
        pred = [int(x) for x in pred]
        trgt = [int(x) for x in trgt]

        #
        # print("Prediction " + str(pred))
        # print("Target " + str(trgt))

        # print("Trying something out")
        pred_indx = [index_to_char[x] for x in pred]
        targ_indx = [index_to_char[x] for x in trgt]
        # print("targ words " + str(targ_indx))
        #
        # print("editditance " + str(editdistance.eval(pred_indx, targ_indx)))

        pred_str = ''.join(pred_indx)
        targ_str = ''.join(targ_indx)

        # print("pred_str " + str(pred_str))
        # print("targ_str " + str(targ_str))

        pred_words = pred_str.split()
        targ_words = targ_str.split()
        #
        # print("pred_words " + str(pred_words))
        # print("targ_words " + str(targ_words))

        errors = editdistance.eval(pred_words, targ_words)

        totalEdits += errors
        totalWords += len(targ_words)
        #
        # predWords = np.split(pred, np.where(pred == spaceIx)[0])
        # predWords = [predWords[0].tostring()] + [predWords[i][1:].tostring() for i in range(1, len(predWords)) if len(predWords[i][1:]) != 0]
        #
        #
        # trgtWords = np.split(trgt, np.where(trgt == spaceIx)[0])
        # trgtWords = [trgtWords[0].tostring()] + [trgtWords[i][1:].tostring() for i in range(1, len(trgtWords))]

        # print("Prediction words " + str(predWords))
        # print("Target words " + str(trgtWords))

        # numEdits = editdistance.eval(predWords, trgtWords)
        # print("PREDICTED ONE " + str(numEdits))
        # exit(1)
        #
        # totalEdits += numEdits
        # totalWords += len(trgtWords)

    wer = totalEdits / totalWords if totalWords > 0 else 0
    return wer