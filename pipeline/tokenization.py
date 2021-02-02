# This is a sample Python script.
from __future__ import unicode_literals
import argparse
import os
from hazm import *
import io
import pyonmttok


# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
def English(file):
    lines = io.open(file, encoding="UTF-8").read()
    tokenized_eng_array = []
    for line in lines.splitlines():
        " ".join(str(x) for x in EnglishTokenizer(line))
        tokenized_eng_array.append(" ".join(str(x) for x in EnglishTokenizer(line)))

    with open("Eng-tok.txt", 'w') as f:
        for item in tokenized_eng_array:
            f.write("%s\n" % item)


def EnglishTokenizer(w):
    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True, case_markup=True)
    tokens, _ = tokenizer.tokenize(w)
    return tokens


def Persian(file):
    lines = io.open(file, encoding="UTF-8").read()
    tokenized_array = []
    for line in lines.splitlines():
        " ".join(str(x) for x in PersianTokenizer(line))
        tokenized_array.append(" ".join(str(x) for x in PersianTokenizer(line)))

    # print(tokenized_array)
    with open("Per-tok.txt", 'w') as f:
        for item in tokenized_array:
            f.write("%s\n" % item)


def PersianTokenizer(w):
    normalizer = Normalizer()  # Persian normalizer instance (Hazm)
    w = normalizer.normalize(w)
    w = word_tokenize(w)
    return w


def main():
    parser = argparse.ArgumentParser(description='Tokenization step for NMT')
    parser.add_argument('--source', '-s', required=True, help="source data path (.txt)")
    parser.add_argument('-srclan', required=True, help="what language is your source data", choices=['English', 'Persian'])
    parser.add_argument('--target', '-t', required=True, help="target data path (.txt)")
    parser.add_argument('-tgtlan', required=True, help="what language is your target data", choices=['English', 'Persian'])
    args = parser.parse_args()

    print("Tokenization Started!")
    print("Step1 ....")
    if args.srclan == 'English':
        English(args.source)
    else:
        English(args.target)

    print("Step2 ....")
    if args.srclan == 'Persian':
        Persian(args.source)
    else:
        Persian(args.target)

    print("Tokenization Done!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
