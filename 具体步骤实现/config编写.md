## config编写
config是一个对模型配置的管理。可以通过_C()创建一个子对象，子对象之间可以嵌套。config文件可以通过.yaml文件更新config具体配置。

具体你可参考代码模板编写。在这个阶段，你只需要按照模板的提示修改即可。

要注意config，config.MODEL一般用于传递模型相关的参数，这一部分最好不要进一步嵌套config的子类，而改成直接把参数赋给config.MODEL对象。例如:

```python

config.MODEL=_C()# 创建一个子对象用于传递模型参数
config.MODEL.TRAIN=.C()
config.MODEL.TRAIN.<params>=<value>
# 改成
config.MODEL=_C()
config.MODEL.<params>=<value>
```