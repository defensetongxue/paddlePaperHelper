## 模型实现

模型实现的时候实际上就是把pytorch对应的语句转化成paddle的结构转化为paddle的语句结构。一下有几个重要的点需要注意。

### paddle和pytorch的api映射表
<https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/08_api_mapping/pytorch_api_mapping_cn.html#pytorch-paddlepaddle-api>

### 模型参数
    针对build_<model>的参数，要尽可能的传入必要的参数，不必要的参数可以直接隐藏在具体的函数类中，在进一步调用具体类的时候采用默认参数。

    必要的参数是指：论文中示例过程中出现改变，以及`dropout`,`depth`等你认为重要参数。
    
    这样，对于不熟悉模型的使用者来说，可以简化参数传入的过程，而对熟练的使用者可以直接对模型代码修改。
### rearrange
    作为一个CV的深度学习库，pytorch在实现的通常采用einpos的rearrange函数，在paddle中我们采用`paddle.transpose`和`paddle.reshape`。具体函数的使用可以参看paddle关于两个函数的文档。
```python
    #b c h w -> b (h w) c.
    B, C, H, W = x.shape
    x = paddle.transpose(x, [0, 2, 3, 1])
    x = paddle.reshape(x, [B, H*W, C])
```

### einsum
    torch.einsum函数对应paddlenlp中的einsum函数，为了让用户减少依赖包的安装，可以使用paddle的transpose、reshape和matmul函数实现相应功能
```python
    # b n h w d, m d -> b n h w m
    B, Nh, H, W, _ = q.shape
    rel_logits = paddle.matmul(q, rel_k.T)
```