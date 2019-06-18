#encoding=utf-8
import tensorflow as tf
import bert_create_model
import modeling
import tokenization
import numpy as np
import os
#不让其使用gpu
os.environ['CUDA_VISIBLE_DEVICES']='-1'
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")
flags.DEFINE_string(
    "bert_config_file", 'checkpointbert/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string("vocab_file", 'checkpointbert/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")
flags.DEFINE_string("model_dir", 'checkpointclass',
                    "bert训练最新的检测点")
labels=['0','1','2','3','4','5','6','7','8']
tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=True)
class Classifier(object):
    def __init__(self):
        #对默认图进行重置
        tf.reset_default_graph()
        gpu_config=tf.ConfigProto()
        gpu_config.gpu_options.allow_growth=True
        self.sess=tf.Session(config=gpu_config)
        self.graph=tf.get_default_graph()
        with self.graph.as_default():
            print("going to restore 11checkpoint")
            self.input_ids_p=tf.placeholder(tf.int32,[None,FLAGS.max_seq_length],name="input_ids")
            self.input_mask_p=tf.placeholder(tf.int32,[None,FLAGS.max_seq_length],name="input_mask")
            bert_config=modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
            #因为下面是通过self.saver的形式加载模型，因此需要先定义需要操作的变量同时需要先构建图
            #这里传入的label是用作构建模型时计算loss时用的，而loss值在随后已用不到
            #随后用到的只有预测值，因此这里的label可以随便传一个值
            (self.total_loss,self.logits,self.trans,self.pred_ids)=bert_create_model.create_model(
                bert_config,False,self.input_ids_p,self.input_mask_p,None,0,len(labels),False)
            self.saver=tf.train.Saver()
            self.saver.restore(self.sess,tf.train.latest_checkpoint(FLAGS.model_dir))
            check_point_path='./restore_ckpt/forpb'
            self.saver.save(self.sess,check_point_path)
if __name__ == '__main__':
    classifier=Classifier()