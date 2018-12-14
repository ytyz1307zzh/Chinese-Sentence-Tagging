import argparse
import Constants

def segment(sentence, tag):
    # apply segmentation on one single sentence using predicted tags
    sentence = list(sentence.strip())
    length = len(sentence)
    tag = list(map(int, tag.strip().split()))

    for i in range(length-2, -1, -1):
        if tag[i] == Constants.BEGIN or tag[i] == Constants.MIDDLE:
            if tag[i+1] == Constants.BEGIN or tag[i+1] == Constants.SINGLE:
                sentence.insert(i+1, ' ')
        elif tag[i] == Constants.END or tag[i] == Constants.SINGLE:
            sentence.insert(i+1, ' ')

    return ''.join(sentence)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_data', required=True, help='sentences without segmentation')
    parser.add_argument('-tags', required=True, help='predicted SBME tags')
    parser.add_argument('-output', required=True, help='where do you want to save the output?')
    opt = parser.parse_args()

    raw_data = open(opt.raw_data, 'r', encoding='utf-8')
    tag_file = open(opt.tags, 'r', encoding='utf-8')
    output_file = open(opt.output, 'w', encoding='utf-8')

    line = 0
    print('[INFO] Begin segmenting')
    while True:
        sentence = raw_data.readline()
        if not sentence:
            break
        tag = tag_file.readline()
        output = segment(sentence, tag)
        output_file.write(output+'\n')
        line += 1
        if line % 100 == 0:
            print('{} lines finished.'.format(line))
        output_file.flush()

    print('[INFO] Segmentation finished.')
    raw_data.close()
    tag_file.close()
    output_file.close()
