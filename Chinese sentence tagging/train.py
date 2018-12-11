import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
from model import make_src_trg_dataset, model
import numpy as np

def train_model(opt, train_data, valid_data, word_embedding):

    BATCH_SIZE = opt.batch
    NUM_EPOCH = opt.epoch

    # datasets used for training and validation
    train_iter = train_data.make_initializable_iterator()
    (train_src, train_src_size), (train_tgt, train_tgt_size) = train_iter.get_next()
    valid_iter = valid_data.make_initializable_iterator()
    (valid_src, valid_src_size), (valid_tgt, valid_tgt_size) = valid_iter.get_next()

    # define the computation graph for training and validation
    train_op,loss_op,accuracy_op = model(train_src, train_src_size, train_tgt, word_embedding, train=True)
    valid_accuracy_op = model(valid_src, valid_src_size, valid_tgt, word_embedding, train=False)
    saver = tf.train.Saver()

    log_tf = open(opt.log+'.train', 'w', encoding='utf-8')
    log_vf = open(opt.log+'valid', 'w', encoding='utf-8')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        highest_accuracy = 0

        for i in range(NUM_EPOCH):
            print("[INFO] Epoch {}:".format(i+1))
            sess.run(train_iter.initializer)
            train_step = 0
            loss_list = []
            train_accu_list = []
            while True:
                try:
                    _, train_loss, train_accuracy = sess.run([train_op, loss_op, accuracy_op])
                    if (train_step * BATCH_SIZE) % 100 == 0:
                        print("{} training samples finished.".format(train_step * BATCH_SIZE))
                    train_step += 1
                    loss_list.append(train_loss)
                    train_accu_list.append(train_accuracy)

                except tf.errors.OutOfRangeError:
                    # save log information
                    aver_loss = sum(loss_list) / len(loss_list)
                    train_aver_accu = sum(train_accu_list) / len(train_accu_list)
                    log_tf.write('epoch: {}, loss: {}, accu: {}%\n'.format(i+1, aver_loss, train_aver_accu*100))
                    break

            sess.run(valid_iter.initializer)
            valid_step = 0
            valid_accu_list = []
            while True:
                try:
                    valid_accuracy = sess.run(valid_accuracy_op)
                    if (valid_step * BATCH_SIZE) % 100 == 0:
                        print("{} validation samples finished.".format(valid_step * BATCH_SIZE))
                    valid_step += 1
                    valid_accu_list.append(valid_accuracy)

                except tf.errors.OutOfRangeError:
                    # save the model with highest accuracy
                    valid_aver_accu = sum(valid_accu_list) / len(valid_accu_list)
                    if valid_aver_accu > highest_accuracy:
                        highest_accuracy = valid_aver_accu
                        saver.save(sess, './'+opt.save_model+'_accu{:.5f}_epoch{}'.format(valid_aver_accu, i+1))
                    # save log information
                    log_vf.write('epoch: {}, accu: {}%\n'.format(i+1, valid_aver_accu*100))
                    break

        print("[INFO] Training finished.")
        log_tf.close()
        log_vf.close()

if __name__ == '__main__':
    # main routine
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-word_embed', required=True, help='pre-trained word embeddings')
    parser.add_argument('-save_model', type=str, default='model', help='path to save the trained model')
    parser.add_argument('-log', type=str, default='log', help='path to saved log information')
    parser.add_argument('-epoch', type=int, default=30, help='how many epochs do you want to train?')
    parser.add_argument('-batch', type=int, default=32, help='the size of a batch')
    opt = parser.parse_args()
    print(opt)

    word_embedding = np.loadtxt(opt.word_embed, dtype=np.float32, encoding='utf-8')
    train_data = make_src_trg_dataset(opt.train_src, opt.train_tgt, BATCH_SIZE)
    valid_data = make_src_trg_dataset(opt.valid_src, opt.valid_tgt, BATCH_SIZE)
    train_model(opt, train_data, valid_data, word_embedding)
