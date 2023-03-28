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

        pred = [int(x) for x in pred]
        trgt = [int(x) for x in trgt]

        pred_indx = [index_to_char[x] for x in pred]
        targ_indx = [index_to_char[x] for x in trgt]

        pred_str = ''.join(pred_indx)
        targ_str = ''.join(targ_indx)

        pred_words = pred_str.split()
        targ_words = targ_str.split()

        errors = editdistance.eval(pred_words, targ_words)

        totalEdits += errors
        totalWords += len(targ_words)

    wer = totalEdits / totalWords if totalWords > 0 else 0
    return wer
