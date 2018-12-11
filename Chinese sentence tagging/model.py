import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants used in the model
VOCAB_SIZE=10002
HIDDEN_SIZE=512
BATCH_SIZE=32
KEEP_PROB=0.8
NUM_TAGS=4
NUM_EPOCH=20

def make_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(lambda string: tf.string_split([tf.convert_to_tensor(string, tf.string)]).values) #将单词切开
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))  #字符串转成整型
    dataset = dataset.map(lambda string: (string, tf.size(string))) #map之后的每个data: (句子tensor,句子长度)
    return dataset

def make_src_trg_dataset(source, target, batch_size): # source为numbered_source.txt的路径,target为numbered_target的路径
    src_data = make_dataset(source)
    trg_data = make_dataset(target)
    dataset=tf.data.Dataset.zip((src_data,trg_data))

    dataset=dataset.shuffle(200000) # shuffle randomly
    padded_shapes=((tf.TensorShape([None]),tf.TensorShape([])),(tf.TensorShape([None]),tf.TensorShape([])))
    dataset=dataset.padded_batch(batch_size,padded_shapes) # pad sentences to the longest length in a batch (padding value: 0)

    # each data in dataset: (([batch,sentence],[batch]),([batch,sentence],[batch])
    return dataset

def model(src_input, src_size, trg_label, word_embedding, train=True):  #src_size=trg_size

    # load pre-trained embeddings
    word_embedding = tf.Variable(tf.convert_to_tensor(word_embedding))
    const_embedding = tf.random_normal([2, 300]) # randomly initialize embeddings for <unk> and <pad>
    word_embedding = tf.concat([const_embedding, word_embedding], axis=0)
    # convert words into vectors, shape=[batch,sentence,embed_size]
    src_embedding=tf.nn.embedding_lookup(word_embedding,src_input)
    # apply dropout if training
    if train:
        src_embedding=tf.nn.dropout(src_embedding,KEEP_PROB)

    # lstm cells
    cell_fw=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell_bw=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

    # bi-rnn
    with tf.variable_scope('rnn', reuse=not train):
        (output_fw,output_bw),_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,src_embedding,
                                                                sequence_length=src_size,dtype=tf.float32)
    # output context vectors, shape=[batch,sentence,2*hidden_size]
    context_rep=tf.concat([output_fw,output_bw],axis=-1)

    max_time=tf.shape(context_rep)[1]  #保存时间维度便于reshape
    nn_input=tf.reshape(context_rep,[-1,2*HIDDEN_SIZE])  #压缩成全连接网络需要的shape=[batch*sentence,2*hidden_size]

    # nn_output.shape=[batch*sentence,num_tags]
    with tf.variable_scope('fc', reuse=not train):
        nn_output=tf.contrib.layers.fully_connected(nn_input,NUM_TAGS)
    output_score=tf.reshape(nn_output,[-1,max_time,NUM_TAGS])  # reshape, shape=[batch,sentence,num_tags]
      #output_score代表每个句子中的每个词对各个class_tag的评分

    #log_likelihood.shape=[batch],transition_params是CRF中递推关系的矩阵,shape=[num_tags,num_tags]
    with tf.variable_scope('crf', reuse=not train):
        log_likelihood,transition_params=tf.contrib.crf.crf_log_likelihood(output_score,trg_label,src_size)
    if train:
        loss=tf.reduce_mean(-log_likelihood)

    #解码评分,decode_tags为预测的tag序列,shape=[batch,sentence], best_score.shape=[batch]
    decode_tags,best_score=tf.contrib.crf.crf_decode(output_score,transition_params,src_size)

    #求准确率,shape=[batch,sentence]
    correct_pred=tf.equal(decode_tags,trg_label)
    accuracy=tf.reduce_mean(tf.reduce_mean(tf.cast(correct_pred,tf.float32),axis=-1))

    #定义优化器
    if train:
        train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        return train_op, loss, accuracy
    else:
        return accuracy
