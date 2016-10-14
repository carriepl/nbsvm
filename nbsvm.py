import os
import pdb
import numpy as np
import argparse
from collections import Counter

def tokenize(sentence, grams):
    words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    return tokens

def build_dict(f, grams):
    dic = Counter()
    for sentence in open(f).xreadlines():
        dic.update(tokenize(sentence, grams))
    return dic

def process_files(file_pos, file_neg, dic, r, outfn, grams, keep_labels):
    output = []
    for beg_line, f in zip(["1", "-1"], [file_pos, file_neg]):
        lines = open(f).readlines()
        
        # Decide which training examples to keep, in a proportion equal to
        # keep_labels.
        nb_keep = int(len(lines) * keep_labels)
        nb_discard = len(lines) - nb_keep
        keep_indices = [1] * nb_keep + [0] * nb_discard
        np.random.seed(0)
        np.random.shuffle(keep_indices)
        
        for l, keep_example in zip(lines, keep_indices):
            if not keep_example:
                continue

            tokens = tokenize(l, grams)
            indexes = []
            for t in tokens:
                try:
                    indexes += [dic[t]]
                except KeyError:
                    pass
            indexes = list(set(indexes))
            indexes.sort()
            line = [beg_line]
            for i in indexes:
                line += ["%i:%f" % (i + 1, r[i])]
            output += [" ".join(line)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()

def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    print "computing r..."
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)
    return dic, r
 
def main(ptrain, ntrain, ptest, ntest, out, liblinear, ngram, train_labels=1.0):
    ngram = [int(i) for i in ngram]
    print "counting..."
    poscounts = build_dict(ptrain, ngram)         
    negcounts = build_dict(ntrain, ngram)     
    
    dic, r = compute_ratio(poscounts, negcounts)
    print "processing files..."
    train_labels = float(train_labels)
    process_files(ptrain, ntrain, dic, r, "train-nbsvm.txt", ngram, train_labels)
    process_files(ptest, ntest, dic, r, "test-nbsvm.txt", ngram, 1.0)
    
    trainsvm = os.path.join(liblinear, "train") 
    predictsvm = os.path.join(liblinear, "predict") 
    os.system(trainsvm + " -s 0 train-nbsvm.txt model.logreg")
    os.system(predictsvm + " -b 1 test-nbsvm.txt model.logreg " + out)
    os.system("rm model.logreg train-nbsvm.txt test-nbsvm.txt")
        
if __name__ == "__main__":
    """
    Usage :

    python nbsvm.py --liblinear /PATH/liblinear-1.96\
        --ptrain /PATH/data/full-train-pos.txt\
        --ntrain /PATH/data/full-train-neg.txt\
        --ptest /PATH/data/test-pos.txt\
        --ntest /PATH/data/test-neg.txt\
         --ngram 123 --out TEST-SCORE
    """

    parser = argparse.ArgumentParser(description='Run NB-SVM on some text files.')
    parser.add_argument('--liblinear', help='path of liblinear install e.g. */liblinear-1.96')
    parser.add_argument('--ptrain', help='path of the text file TRAIN POSITIVE')
    parser.add_argument('--ntrain', help='path of the text file TRAIN NEGATIVE')
    parser.add_argument('--ptest', help='path of the text file TEST POSITIVE')
    parser.add_argument('--ntest', help='path of the text file TEST NEGATIVE')
    parser.add_argument('--out', help='path and fileename for score output')
    parser.add_argument('--ngram', help='N-grams considered e.g. 123 is uni+bi+tri-grams')
    parser.add_argument('--train_labels', help='fraction (float) of training labels to keep')
    args = vars(parser.parse_args())

    main(**args)
