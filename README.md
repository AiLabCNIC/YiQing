###针对不同任务，更改以下内容
#### 数据处理 
1. data_preprocess.py
   *  plot_text_length: 可视化待训练数据的长度（确定参数 -- max_sequence_length）
   *  cut_fold: 创建训练、验证、测试的数据
2. clean_data.py
   * 数据清洗

#### **数据载入**（必须）
1. dataset.py
    * 待训练的特征、进而token的组成
    * 标签

#### 运行
1. args.py: 分类数量
2. model.py: 模型构建