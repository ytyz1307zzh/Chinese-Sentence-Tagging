# -*- coding: UTF-8 -*-
import argparse
import Constants
from tqdm import tqdm

def char_split(data_path, store_path):
    # split text into characters (divided by whitespace)
    data_file = open(data_path, 'r', encoding='utf-8')
    output_file = open(store_path, 'w', encoding='utf-8')

    print('[INFO] Start splitting sentences into separate characters...')
    for line in tqdm(data_file, ascii=True):
        text = line.strip()
        text = [word for word in text if word != ' ']
        text = ' '.join(text)
        print(text, file=output_file)
    print('[INFO] Finished.')

    data_file.close()
    output_file.close()

def char2num(file_path, vocab_path, store_path):
    # transfer Chinese characters into numbers according to the order in vocabulary
    vocab_file = open(vocab_path, 'r', encoding='utf-8')
    vocab = vocab_file.readline()
    vocab = vocab.strip().split()

    file = open(file_path, 'r', encoding='utf-8')
    store_file = open(store_path, 'w' ,encoding='utf-8')

    print('[INFO] Start transferring characters into numbers...')
    for line in tqdm(file, ascii=True):
        sentence = line.strip().split()
        # words out of vocabulary are replaced by <unk>
        sentence = [str(vocab.index(word)) if word in vocab else str(Constants.UNK) for word in sentence]
        sentence = ' '.join(sentence)
        print(sentence, file=store_file)
    print('[INFO] Finished.')

    vocab_file.close()
    store_file.close()
    file.close()

def SBMEtag(file_path, output_path):
    # convert the target answers to SBME tags
    file = open(file_path, 'r', encoding='utf-8')
    output_file = open(output_path, 'w', encoding='utf-8')

    print('[INFO] Start transferring characters into SBME tags...')
    for line in tqdm(file, ascii=True):
        tags = []
        sentence = line.strip().split()
        for word in sentence:
            length = len(word)
            if length == 1:
                tags.append(str(Constants.SINGLE))
            elif length > 1:
                tags.append(str(Constants.BEGIN))
                length -= 2
                while length > 0:
                    tags.append(str(Constants.MIDDLE))
                    length -= 1
                tags.append(str(Constants.END))
        print(' '.join(tags), file=output_file)
    print('[INFO] Finished.')

    file.close()
    output_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_mode', type=str, choices=['source', 'answer'], default='src', required=True,
                        help='-source should be the source text, -answer should be the answer text')
    parser.add_argument('-src', required=True, help='raw text file')
    parser.add_argument('-output', required=True, help='output text file')
    parser.add_argument('-vocab', help='vocab_file')
    parser.add_argument('-temp_file', help='temporary file to save split result (optional)')

    opt = parser.parse_args()
    if opt.data_mode == 'source':
        if not opt.temp_file:
            opt.temp_file = './temp.txt'
        if not opt.vocab:
            print('[ERROR] Vocabulary is not provided.')
        char_split(opt.src, opt.temp_file)
        char2num(opt.temp_file, opt.vocab, opt.output)
    elif opt.data_mode == 'answer':
        SBMEtag(opt.src, opt.output)


