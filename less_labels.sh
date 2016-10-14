for ngram in '1' '12' '123'
do
    for keep_labels in 0.01 0.05 0.1 0.25 0.5 1.0
    do
        command="python nbsvm.py --liblinear ../nbsvm_run/liblinear-1.96 --ptrain ../nbsvm_run/data/train-pos.txt --ntrain ../nbsvm_run/data/train-neg.txt --ptest ../nbsvm_run/data/test-pos.txt --ntest ../nbsvm_run/data/test-neg.txt --ngram $ngram --out NBSVM-TEST-$ngram-$keep_labels --train_labels=$keep_labels > exp_log-$ngram-$keep_labels.txt"
        eval $command
    done
done
