# BLEU for segment by segment with arguments
# Run this file from CMD/Terminal
# Example Command: python3 compute-bleu-sentence-args.py test_file_name.txt mt_file_name.txt

import sys
import sacrebleu
from sacremoses import MosesDetokenizer
md = MosesDetokenizer(lang='en')

target_test = sys.argv[1]  # Test file argument
target_pred = sys.argv[2]  # MTed file argument

# Open the test dataset human translation file and detokenize the references
refs = []

with open(target_test) as test:
    for line in test: 
        line = line.strip().split() 
        line = md.detokenize(line) 
        refs.append(line)
    
print("Reference 1st sentence:", refs[0])

# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open(target_pred) as pred:  
    for line in pred: 
        line = line.strip().split() 
        line = md.detokenize(line) 
        preds.append(line)

# Calculate BLEU for sentence by sentence and save the result to a file
with open("bleu-" + target_pred + ".txt", "w+") as output:
    for line in zip(refs,preds):
        test = line[0]
        pred = line[1]
        print(test, "\t--->\t", pred)
        bleu = sacrebleu.sentence_bleu(pred, [test], smooth_method='exp')
        print(bleu.score, "\n")
        output.write(str(bleu.score) + "\n")
