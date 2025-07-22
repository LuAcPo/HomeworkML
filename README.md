redata.py 数据集下载\
openCora.py 数据预处理\
data_pre.py NewGraph操作\
models.py 模型\
train.py 训练+部分消融
```python
data,mask,graph,labels = GetData(temperature)#normal
```
```python
data,_,_,labels = GetData(temperature)
mask,graph = getrawData()#only labelprompt
```
alb.py
```python
data,mask,graph,labels = GetData(temperature)#only NewGraph
```
```python
data,_,_,labels = GetData(temperature)
mask,graph = getrawData()#none
```
