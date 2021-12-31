import paddle
import torch
import numpy as np
import os

# 引入paddle和pytorch模型，这是你需要根据自己的模型更改的部分
from pytorch_file import  py_model
from paddle_file import  pa_model

def print_model_named_params(model):
    print('----------------------------------')
    for name, param in model.named_parameters():
        print(name, param.shape)
    print('----------------------------------')

def print_model_named_buffers(model):
    print('----------------------------------')
    for name, param in model.named_buffers():
        print(name, param.shape)
    print('----------------------------------')

# 以下是针对cvt的torch模型权重到paddle模型权重的映射函数
def torch_to_paddle_mapping():
    # mapping会存储全部网络权重映射的tuple
    mapping = [('stage2.cls_token', 'stages.2.cls_token')]

    # 根据网络的深度，如果有重复的映射则写入循环中
    depths = [2, 2, 20]
    num_stages = len(depths)
    for stage_idx in range(num_stages):
        pp_s_prefix = f'stages.{stage_idx}'
        th_s_prefix = f'stage{stage_idx}'
        layer_mapping = [
            (f'{th_s_prefix}.patch_embed.proj', f'{pp_s_prefix}.patch_embed.proj'),
            (f'{th_s_prefix}.patch_embed.norm', f'{pp_s_prefix}.patch_embed.norm'),
        ] 
        mapping.extend(layer_mapping)

        for block_idx in range(depths[stage_idx]):
            th_b_prefix = f'{th_s_prefix}.blocks.{block_idx}'
            pp_b_prefix = f'{pp_s_prefix}.blocks.{block_idx}'
            layer_mapping = [
                (f'{th_b_prefix}.norm1', f'{pp_b_prefix}.norm1'),
                (f'{th_b_prefix}.attn.conv_proj_q.conv', f'{pp_b_prefix}.attn.conv_proj_q.0'),
                (f'{th_b_prefix}.attn.conv_proj_q.bn', f'{pp_b_prefix}.attn.conv_proj_q.1'),
                (f'{th_b_prefix}.attn.conv_proj_k.conv', f'{pp_b_prefix}.attn.conv_proj_k.0'),
                (f'{th_b_prefix}.attn.conv_proj_k.bn', f'{pp_b_prefix}.attn.conv_proj_k.1'),
                (f'{th_b_prefix}.attn.conv_proj_v.conv', f'{pp_b_prefix}.attn.conv_proj_v.0'),
                (f'{th_b_prefix}.attn.conv_proj_v.bn', f'{pp_b_prefix}.attn.conv_proj_v.1'),
                (f'{th_b_prefix}.attn.proj_q', f'{pp_b_prefix}.attn.proj_q'),
                (f'{th_b_prefix}.attn.proj_k', f'{pp_b_prefix}.attn.proj_k'),
                (f'{th_b_prefix}.attn.proj_v', f'{pp_b_prefix}.attn.proj_v'),
                (f'{th_b_prefix}.attn.proj', f'{pp_b_prefix}.attn.proj'),
                (f'{th_b_prefix}.norm2', f'{pp_b_prefix}.norm2'),
                (f'{th_b_prefix}.mlp.fc1', f'{pp_b_prefix}.mlp.fc1'),
                (f'{th_b_prefix}.mlp.fc2', f'{pp_b_prefix}.mlp.fc2'),
            ]
            mapping.extend(layer_mapping)

    mapping.extend([
        ('norm', 'norm'),
        ('head', 'head')])
    return mapping

def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, no_transpose=False):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].data.numpy()
        if len(value.shape) == 2:
            if not no_transpose:
                value = value.transpose((1, 0))
        pd_params[pd_name].set_value(value)
    
    # 1. get paddle and torch model parameters
    pd_params = {}
    th_params = {}
    for name, param in paddle_model.named_parameters():
        pd_params[name] = param
    for name, param in torch_model.named_parameters():
        th_params[name] = param

    for name, param in paddle_model.named_buffers():
        pd_params[name] = param
    for name, param in torch_model.named_buffers():
        th_params[name] = param
    
    # 2. get name mapping pairs
    mapping = torch_to_paddle_mapping()
    # 3. set torch param values to paddle params: may needs transpose on weights
    for th_name, pd_name in mapping:
        if th_name in th_params.keys(): # nn.Parameters
            if th_name.endswith('relative_position_bias_table'):
                _set_value(th_name, pd_name, no_transpose=True)
            else:
                _set_value(th_name, pd_name)
        else: # weight & bias
            # 你需要根据你的模型权重加入以下部分，针对th_params中没有的权重关键词进行单独映射设置
            # 以下是cvt实例
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_mean' in th_params.keys():
                th_name_b = f'{th_name}.running_mean'
                pd_name_b = f'{pd_name}._mean'
                _set_value(th_name_b, pd_name_b)

            if f'{th_name}.running_var' in th_params.keys():
                th_name_b = f'{th_name}.running_var'
                pd_name_b = f'{pd_name}._variance'
                _set_value(th_name_b, pd_name_b)

    return paddle_model

def main():
    # 导入你自己的paddle模型
    paddle_model = pa_model()
    paddle_model.eval()
    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')

    # 导入你自己的pytorch模型
    torch_model = py_model(pretrained=True)
    torch_model.eval()
    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    # check correctness
    x = np.random.randn([2, 3, 224, 224]).astype('float32')
    x_paddle = paddle.to_tensor(x)
    x_torch = torch.Tensor(x).to(device)


    out_torch = torch_model(x_torch)
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    print('|||||||||||||||||||||||||||||||||||||||||||||||||||')
    out_paddle = paddle_model(x_paddle)

    out_torch = out_torch.data.cpu().numpy()
    out_paddle = out_paddle.cpu().numpy()

    print(out_torch.shape, out_paddle.shape)
    print(out_torch[0, 0:20])
    print(out_paddle[0, 0:20])
    assert np.allclose(out_torch, out_paddle, atol = 1e-4)

    # save weights for paddle model
    # 加入你自己存储模型的位置，存储文件后缀固定为pdparams
    model_path = os.path.join('your_save_path/model.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()