# Corpus WER

# WER score for the whole corpus
# Run this file from CMD/Terminal
# Example Command: python3 corpus-wer.py test_file_name.txt mt_file_name.txt

import sys
from jiwer import wer


target_test = sys.argv[1]	#  Test file argument
target_pred = sys.argv[2]	#  MTed file argument


# Open the test dataset human translation file
with open(target_test) as test:
    refs = test.readlines()

#print("Reference 1st sentence:", refs[0])

# Open the translation file by the NMT model
with open(target_pred) as pred:
    preds = pred.readlines()

wer_file = "wer-" + target_pred + ".txt"

# Calculate WER for the whole corpus
wer_score = wer(refs, preds, standardize=True)    # "standardize" expands abbreviations
print("WER Score:", wer_score)
