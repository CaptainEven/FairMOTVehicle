# FairMOTVehicle
A fork of FairMOT used to do vehicle MOT.

#### 车辆跟踪，效果如下，此测试未经过训练(Results of vehicle mot is as follows, the video seq has not been trained)： </br>
![image](https://github.com/CaptainEven/FairMOTVehicle/blob/master/results/frame/result_vehicle.gif) 
</br>
#### 使用UA-DETRAC公开数据集训练FairMOT
UA_DETRAC是一个公开的车辆跟踪数据集, 共8万多张训练数据集，每一张图的每一辆车都经过了精心的标注。</br>
[UA-DETRAC](http://detrac-db.rit.albany.edu/) </br>
</br>
#### 训练方法
##### (1). 使用gen_labels_detrac.py脚本预处理原始的训练数据
##### (2). 编写ccfg文件
##### (3). 修改opts.py文件
