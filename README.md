# image-caption
#### image caption实际上是一个看图说话的任务，即输入一张图，输出图片的描述。此项目数据集来源于[COCO2014](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)，包含80000多张图片。此项目采用tf.keras进行建模，模型很简单，适合初学者进行学习、练习    

#### TensorFlow官方力推、GitHub爆款项目：用Attention模型自动生成图像字幕
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb  
官方给的代码使用的是tensorflow的eager模式，可自行查阅参考。

### Run
1. python data_hepler.py  
划分数据，使用了20000张图片  

2. python image_caption_keras.py  
开始训练，并保存模型  

### Tutorial Overview
1、数据划分、文本清洗、构建数据集，标签需要自己构造。  
比如：对于two women stand on each side of the elephant来说，重新构造数据的方法：  
| image_feature | caption | label |
| :------| ------: | :------: |
| photo | <start> | two |
| photo | <start> two | women |
| photo | <start> two women | stand |
| photo | <start> two women stand | on |
| photo | ... | ... |
| photo | <start> two women stand on each side of the elephant | <end> |  
每次将同一张图片和该图片描述前面的词输入模型，模型的输出是描述的后一个词  
2、模型  
3、测试  
查看result.ipynb文件
