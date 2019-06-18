#encoding=utf-8
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
#和tf.graph()所有相关的函数
os.environ['CUDA_VISIBLE_DEVICES']='-1'
dir=os.path.dirname(os.path.realpath(__file__))
def freeze_graph(model_folder,output_filename):
    '''
    :param model_folder: 检测点路径
    :param output_filename: pb文件的路径
    :return:
    '''
    checkpoint=tf.train.get_checkpoint_state(model_folder)
    input_checkpoint=checkpoint.model_checkpoint_path
    absolute_model_folder="\\".join(input_checkpoint.split('\\')[:-1])
    output_graph=absolute_model_folder+output_filename
    #freeze之前必须明确哪个是输出节点，也就是模型预测结果的节点
    #输出节点可以看我们模型的定义
    #只有定义了输出节点，freeze才会把得到输出节点所必要的节点都保存下来，或者哪些节点可以丢弃
    #所以，output_node_names必须根据不同的网络进行修改
    output_node_names="loss/Softmax"
    clear_devices=True
    #我们清除这些设备，以允许TensorFlow控制它想要计算操作的负载
    #加载图和数据
    saver=tf.train.import_meta_graph(input_checkpoint+'.meta',clear_devices=clear_devices)
    #加载默认图
    graph=tf.get_default_graph()
    #返回一个图的序列化的GraphDef表示序列化的GraphDef可以导入至另一个图中(使用 import_graph_def())
    input_graph_def=graph.as_graph_def()
    #通过import_meta_graph的形式既加载了图又加载了数据，随后便可进行固化
    #把参数和图固化并封装成pb
    with tf.Session() as sess:
        saver.restore(sess,input_checkpoint)
        #使用tf内置的辅助函数把变量导出为常量
        output_graph_def=graph_util.convert_variables_to_constants(sess,input_graph_def,output_node_names.split(","))
        #把图序列化并转储到文件系统
        with tf.gfile.GFile(output_graph,'wb') as f:
            f.write(output_graph_def.SerializeToString())
            #output_graph_def.node获取所有节点的名称
        print('%d ops in final graph.' % len(output_graph_def.node))
if __name__ == '__main__':
    #
    freeze_graph('./restore_ckpt','intention.pb')