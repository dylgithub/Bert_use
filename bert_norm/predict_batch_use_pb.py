#encoding=utf-8
import tensorflow as tf
import tokenization
import numpy as np
from tensorflow.python.platform import gfile
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "bert_config_file", 'checkpointbert/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string("vocab_file", 'checkpointbert/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string("model_dir", 'checkpointclass',
                    "bert训练最新的检测点")
labels=['0','1','2','3','4','5','6','7','8']
tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)
class Classifier(object):
    def __init__(self,pb_path):
        self.sess=tf.Session()
        with gfile.FastGFile(pb_path,'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(graph_def,name='')#导入计算图
        #需要一个初始化的过程
        self.sess.run(tf.global_variables_initializer())
        #其中input_ids是根据封pb时候的命名改变的,.outputs[0]是获得这个operation的第0个输出tensor
        self.input_ids_p =self.sess.graph.get_operation_by_name('input_ids').outputs[0]
        self.input_mask_p =self.sess.graph.get_operation_by_name('input_mask').outputs[0]
        self.pred_ids =self.sess.graph.get_operation_by_name('loss/Softmax').outputs[0]
    def batch_iter(self,id,mask,batch_size=64):
        data_len = len(id)
        num_batch = int((data_len - 1) / batch_size) + 1
        # 打乱数据，若数据已打乱则不需要
        # indices = np.random.permutation(np.arange(data_len))
        # x_shuffle = x[indices]
        # y_shuffle = y[indices]
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield id[start_id:end_id],mask[start_id:end_id]
    def predict(self,sentence_list):
        input_ids, input_mask, segment_ids, label_ids=convert(sentence_list)
        res_list=[]
        for input_ids,input_mask in self.batch_iter(input_ids,input_mask):
            feed_dict={
                self.input_ids_p:input_ids,
                self.input_mask_p:input_mask
            }
            pred_ids_result=self.sess.run([self.pred_ids],feed_dict)
            #这里是类别标签的中文名
            bc=['0','1']
            res_ids=list(np.argmax(pred_ids_result[0],axis=-1))
            temp_res_list=[bc[i] for i in res_ids]
            res_list.extend(temp_res_list)
        return res_list
def convert(sentence_list):
    input_ids_list=[]
    input_mask_list=[]
    segment_ids_list=[]
    label_ids_list=[]
    for i,line in enumerate(sentence_list):
        feature = convert_single_example(0,line,labels, FLAGS.max_seq_length, tokenizer)
        input_ids_list.append(feature.input_ids)
        input_mask_list.append(feature.input_mask)
        segment_ids_list.append(feature.input_mask)
        label_ids_list.append(feature.label_id)
    input_ids = np.reshape(input_ids_list, (-1, FLAGS.max_seq_length))
    input_mask = np.reshape(input_mask_list, (-1, FLAGS.max_seq_length))
    segment_ids = np.reshape(segment_ids_list, (-1, FLAGS.max_seq_length))
    label_ids=np.array(label_ids_list)
    return input_ids, input_mask, segment_ids, label_ids
class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
#将样本转换成features,这里有改动的地方，对于单文本分类只需传入一个句子
#不需要和源码一样example.text_a，example.text_b
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  #样本向字id的转换
  tokens_a = tokenizer.tokenize(example)
  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)
  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  # 所有类别标签的第一个类别标签
  label_id = label_map['0']
  # 创建InputFeatures的一个实例化对象并返回
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature
if __name__ == '__main__':
    classifier=Classifier('intention.pb')
    test_list=['a','b']
    res_list=classifier.predict(test_list)