# 关于

“基于自注意力路由胶囊网络的交通流预测”论文的pytorch实现

# 配置

在config.yaml中可以进行详细的参数设置

# 数据集

使用了METR-LA数据集。执行以下代码生成数据集：

```python
python generate_data.py	
```

# 依赖

安装依赖：

```
pip install -r requirements.txt
```

# 训练

设置完配置文件后，执行以下代码进行训练：

```python
python train.py
```



