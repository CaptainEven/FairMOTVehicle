# FairMOTVehicle
A fork of FairMOT used to do vehicle MOT(multi-object tracking).
You can refer to origin fork </br>
[FairMOT](https://github.com/ifzhang/FairMOT)
#### 车辆跟踪，效果如下，此测试未经过训练(Results of vehicle mot is as follows, the video seq has not been trained)： </br>
![image](https://github.com/CaptainEven/FairMOTVehicle/blob/master/results/frame/result_vehicle.gif) 
![image](https://github.com/CaptainEven/FairMOTVehicle/blob/master/results/frame/result1.jpg)
![image](https://github.com/CaptainEven/FairMOTVehicle/blob/master/results/frame/result2.jpg)
</br>
#### 使用UA-DETRAC公开数据集训练FairMOT(Using UA-DETRAC as training dataset for vehicle tracking)
UA_DETRAC是一个公开的车辆跟踪数据集, 共8万多张训练数据集，每一张图的每一辆车都经过了精心的标注。</br>
[UA-DETRAC](http://detrac-db.rit.albany.edu/) </br>

#### 训练方法(具体调用时，根据服务器目录, 修改自定义路径)
##### (1). 使用gen_labels_detrac.py脚本预处理原始的训练数据(Call gen_labels_detrac.py to prepare UA-DETRAC for training)
###### <1>. 调用preprocess函数创建用于FairMOT的标准训练数据目录
###### <2>. 调用核心函数, gen_labels函数，解析UA-DETRAC的xml格式标签文件转换成FairMOT格式的标签文件，生成txt标签文件
###### <3>. 调用gen_dot_train_file函数，生成用于训练的.train文件
##### (2). 编写json格式的cfg文件./src/lib/cfg/detrac.json
##### (3). 修改opts.py文件，修改训练参数，开始训练
###### <1>. 修改--load_model参数, 选择一个断点模型, 如 ctdet_coco_dla_2x.pth, 从这个预训练模型开始训练
###### <2>. 修改----data_cfg参数, 选择训练、测试数据，如 ../src/lib/cfg/detrac.json
###### <3>. python or python3 ./src/train.py启动训练进程即可。
