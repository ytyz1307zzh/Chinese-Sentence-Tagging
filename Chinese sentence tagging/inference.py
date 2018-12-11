import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from test import make_src_dataset, inference
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', required=True)
    parser.add_argument('-model', required=True, help='path to pre-trained model')
    parser.add_argument('-output', required=True, help='path to store the output')
    parser.add_argument('-batch', type=int, default=32, required=True, help='size of a batch')
    opt = parser.parse_args()

    dataset=make_src_dataset(opt.src, opt.batch)
    iterator=dataset.make_initializable_iterator()
    src_input,src_size=iterator.get_next()

    #定义计算图
    sequence_op = inference(src_input,src_size)
    saver = tf.train.Saver()
    result = []

    with tf.Session() as sess:
        saver.restore(sess, opt.model)
        sess.run(iterator.initializer)
        print('[INFO] Start inference process...')
        while True:
            step = 0
            try:
                sequence = sess.run(sequence_op).tolist()
                step += 1
                if (step * opt.batch) % 100 ==0:
                    print('{} test instances finished.'.format(step*opt.batch))
                for seq in sequence:
                    result.append(seq)
            except tf.errors.OutOfRangeError:
                break
        with open(opt.output, 'w', encoding='utf-8') as file:
            for seq in result:
                length = len(seq) - 1
                while length > -1 and seq[length] == 0:
                    seq.pop()
                    length -= 1
                file.write(' '.join(map(str, seq))+'\n')
                print('[INFO] Inference finished.')
