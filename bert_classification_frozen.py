import json
import os
import sys
import modeling
import tensorflow as tf
tf.reset_default_graph()

# 设置信息
max_seq_len = 64
bert_model_dir = '../pre_model_wwm/publish/'
is_training=False
segment_ids=None
labels= ['P','M','N']
num_labels= len(labels)
model_dir = '../emotion_datasets/bert_output/'
pb_file = '../emotion_datasets/bert_output/2/'


def create_classification_model(bert_config, input_ids, input_mask, num_labels):
    '''Bert-classifier模型搭建
    
       input
          input_ids：输入1，shape(?,max_seq_len)
          input_mask：输入2，shape(?,max_seq_len)
          bert_config：bert配置信息，json
          num_labels：分类数，int
          
       return
          probabilities：输出，shape(?,num_labels)
    '''
    # 创建bert模型
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=None,
    )
    
    #获取输出层信息，ckpt里边包含了
    embedding_layer = model.get_sequence_output() #获取encode的最后的output
    output_layer = model.get_pooled_output() #shape=(?, 768)
    hidden_size = output_layer.shape[-1].value #获取隐含层数
    
    #新建输出层W,B  ckpt文件里已经包含了数据以及shape，此处新建一个空的即可
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)) #或者tf.zeros_initializer()

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())
    
    #创建softmax分类器，这个不包含在ckpt里，需要自建，实际上和classifier脚本里一致
    with tf.variable_scope("loss"):
        logits = tf.matmul(output_layer, output_weights, transpose_b=True) # (?, 768) * (3, 768).T = (?, 3)
        logits = tf.nn.bias_add(logits, output_bias) #(?, 3)
        probabilities = tf.nn.softmax(logits, axis=-1) #(?, 3)

    return probabilities
    
# 模型搭建
graph = tf.Graph().as_default() #新建图

# 搭建模型，输入input_ids, input_mask；输出probabilities
# 张量name分别对应'input_ids'、'input_mask'、'pred_prob'

input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')

# 加载bert模型参数
bert_config = modeling.BertConfig.from_json_file(os.path.join(bert_model_dir, 'bert_config.json'))

# 构建model
probabilities = create_classification_model(bert_config, input_ids, input_mask, num_labels)
probabilities = tf.identity(probabilities, 'pred_prob')

# 新建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())


#载入ckpt数据
saver = tf.train.Saver()
latest_checkpoint = tf.train.latest_checkpoint(model_dir)
saver.restore(sess,latest_checkpoint )

# 载入测试数据
import pickle
with open('data.pkl', 'rb') as f1:
    input_token = pickle.load(f1)
input_token.keys()

# run
result_predict = sess.run(probabilities,feed_dict={input_ids: input_token['input_ids'],
                                  input_mask: input_token['input_mask']})
                                  
# 生成pb模型
# 通过sess保存,outputs自定义信号如下，为符合平台在线测试用的
tf.saved_model.simple_save(sess,export_dir=pb_file,
                           inputs={"input_ids": input_ids,"input_mask":input_mask},outputs={"pred_prob": probabilities})
                           
# 压缩
! cd /clever/code/bert_a/source/bert-master/emotion_datasets/bert_output/;pwd;zip -r 2.zip 2

# result_predict_index = result_predict.argmax(axis=1)
flag = {0:'P',1:'M',2:'N'}
result_predict_finally = [flag[i] for i in result_predict_index]
result_predict_finally
