import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

VOCAB_SIZE=10002
HIDDEN_SIZE=512
KEEP_PROB=0.8
NUM_TAGS=4

def make_dataset(file_path):
    dataset=tf.data.TextLineDataset(file_path)
    dataset=dataset.map(lambda string: tf.string_split([tf.convert_to_tensor(string,tf.string)]).values) #将单词切开
    dataset=dataset.map(lambda string: tf.string_to_number(string,tf.int32))  #字符串转成整型
    dataset=dataset.map(lambda string: (string,tf.size(string))) #map之后的每个data: (句子tensor,句子长度)
    return dataset

def make_src_dataset(source,batch_size): #source为numbered_source.txt的路径,target为numbered_target的路径

    dataset=make_dataset(source)
    padded_shapes=(tf.TensorShape([None]),tf.TensorShape([]))
    dataset=dataset.padded_batch(batch_size,padded_shapes)

    #dataset中每条数据shape:(([batch,sentence],[batch])
    return dataset

def inference(src_input,src_size):  #src_size=trg_size

    # get embeddings
    word_embedding = tf.get_variable('Variable',shape=[VOCAB_SIZE-2,300],dtype=tf.float32)
    const_embedding = tf.random_normal([2, 300])
    word_embedding = tf.concat([const_embedding, word_embedding], axis=0)
    # convert to vectors, shape=[batch,sentence,hidden_size]
    src_embedding=tf.nn.embedding_lookup(word_embedding,src_input)

    cell_fw=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell_bw=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

    with tf.variable_scope('rnn'):
        (output_fw,output_bw),_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,src_embedding,
                                                  sequence_length=src_size,dtype=tf.float32)
    # context vector, shape=[batch,sentence,2*hidden_size]
    context_rep=tf.concat([output_fw,output_bw],axis=-1)

    max_time=tf.shape(context_rep)[1]  #保存时间维度便于reshape
    nn_input=tf.reshape(context_rep,[-1,2*HIDDEN_SIZE])  #压缩成全连接网络需要的shape=[batch*sentence,2*hidden_size]

    # nn_output.shape=[batch*sentence,num_tags]
    with tf.variable_scope('fc'):
        nn_output=tf.contrib.layers.fully_connected(nn_input,NUM_TAGS)
    output_score=tf.reshape(nn_output,[-1,max_time,NUM_TAGS])  # reshape, shape=[batch,sentence,num_tags]
      # output_score代表每个句子中的每个词对各个class_tag的评分

    #transition_params是CRF中递推关系的矩阵,shape=[num_tags,num_tags]
    with tf.variable_scope('crf'):
        transition_params=tf.get_variable(name='transitions',dtype=tf.float32,shape=[NUM_TAGS,NUM_TAGS])

    decode_tags,_ = tf.contrib.crf.crf_decode(output_score,transition_params,src_size)

    return decode_tags

