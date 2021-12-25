import paddle
import torch
import numpy as np
from botnet import botnet50
from botnet_torch import botnet50_torch
import os
import pdb

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

def torch_to_paddle_mapping():
    mapping = [
        ('0', '0'),
        ('1', '1'),
    ]
    
    num_stages = 8
    num_blocks = [0, 0, 0, 0, 3, 4, 6, 3]
    for stage_idx in range(num_stages):
        if stage_idx == 7:
            mapping.extend([
                (f'7.net.0.shortcut.0', f'7.net.0.shortcut.0'),
                (f'7.net.0.shortcut.1', f'7.net.0.shortcut.1'),
            ])
        for block_idx in range(num_blocks[stage_idx]):
            if num_blocks[stage_idx] == 0:
                break
            if stage_idx == 7:
                pp_prefix = f'{stage_idx}.net.{block_idx}'
                th_prefix = f'{stage_idx}.net.{block_idx}'
                layer_mapping = [
                    (f'{th_prefix}.net.0', f'{pp_prefix}.net.0'),
                    (f'{th_prefix}.net.1', f'{pp_prefix}.net.1'),
                    (f'{th_prefix}.net.3.to_qk', f'{pp_prefix}.net.3.to_qk'),
                    (f'{th_prefix}.net.3.to_v', f'{pp_prefix}.net.3.to_v'),
                    (f'{th_prefix}.net.3.pos_emb', f'{pp_prefix}.net.3.pos_emb'),
                    (f'{th_prefix}.net.5', f'{pp_prefix}.net.5'),
                    (f'{th_prefix}.net.7', f'{pp_prefix}.net.7'),
                    (f'{th_prefix}.net.8', f'{pp_prefix}.net.8'),
                ]
                mapping.extend(layer_mapping)
            else:
                pp_prefix = f'{stage_idx}.{block_idx}'
                th_prefix = f'{stage_idx}.{block_idx}'
                layer_mapping = [
                    (f'{th_prefix}.conv1', f'{pp_prefix}.conv1'),
                    (f'{th_prefix}.bn1', f'{pp_prefix}.bn1'),
                    (f'{th_prefix}.conv2', f'{pp_prefix}.conv2'),
                    (f'{th_prefix}.bn2', f'{pp_prefix}.bn2'),
                    (f'{th_prefix}.conv3', f'{pp_prefix}.conv3'),
                    (f'{th_prefix}.bn3', f'{pp_prefix}.bn3'),
                ]
                mapping.extend(layer_mapping)
                if block_idx == 0:
                    mapping.extend([
                        (f'{th_prefix}.downsample.0', f'{pp_prefix}.downsample.0'),
                        (f'{th_prefix}.downsample.1', f'{pp_prefix}.downsample.1'),
                    ])
    mapping.extend([('10', '10')])
    return mapping

def convert(torch_model, paddle_model):
    def _set_value(th_name, pd_name, no_transpose=False):
        th_shape = th_params[th_name].shape
        pd_shape = tuple(pd_params[pd_name].shape) # paddle shape default type is list
        #assert th_shape == pd_shape, f'{th_shape} != {pd_shape}'
        print(f'set {th_name} {th_shape} to {pd_name} {pd_shape}')
        value = th_params[th_name].data.numpy()
        if len(value.shape) == 2 and th_shape != pd_shape:
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
            if f'{th_name}.weight' in th_params.keys():
                th_name_w = f'{th_name}.weight'
                pd_name_w = f'{pd_name}.weight'
                _set_value(th_name_w, pd_name_w)

            if f'{th_name}.bias' in th_params.keys():
                th_name_b = f'{th_name}.bias'
                pd_name_b = f'{pd_name}.bias'
                _set_value(th_name_b, pd_name_b)
            
            if f'{th_name}.running_mean' in th_params.keys():
                th_name_rm = f'{th_name}.running_mean'
                pd_name_rm = f'{pd_name}._mean'
                _set_value(th_name_rm, pd_name_rm)

            if f'{th_name}.running_var' in th_params.keys():
                th_name_rv = f'{th_name}.running_var'
                pd_name_rv = f'{pd_name}._variance'
                _set_value(th_name_rv, pd_name_rv)

            if f'{th_name}.rel_height' in th_params.keys():
                th_name_rh = f'{th_name}.rel_height'
                pd_name_rh = f'{pd_name}.rel_height'
                _set_value(th_name_rh, pd_name_rh)
            
            if f'{th_name}.rel_width' in th_params.keys():
                th_name_rw = f'{th_name}.rel_width'
                pd_name_rw = f'{pd_name}.rel_width'
                _set_value(th_name_rw, pd_name_rw)

    return paddle_model

def main():
    paddle_model = botnet50()
    paddle_model.eval()
    print_model_named_params(paddle_model)
    print_model_named_buffers(paddle_model)

    print('+++++++++++++++++++++++++++++++++++')
    device = torch.device('cpu')
    torch_model = botnet50_torch(pretrained=True)
    torch_model.eval()
    print_model_named_params(torch_model)
    print_model_named_buffers(torch_model)

    # convert weights
    paddle_model = convert(torch_model, paddle_model)

    a = torch_model.state_dict()
    b = paddle_model.state_dict()

    # check correctness
    x = np.ones([2, 3, 224, 224]).astype('float32')
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
    a = torch_model.state_dict()
    b = paddle_model.state_dict()
    assert np.allclose(out_torch, out_paddle, atol = 1e-4)

    # save weights for paddle model
    model_path = os.path.join(r'D:\app\vscode\py\paddle\test\botnet50.pdparams')
    paddle.save(paddle_model.state_dict(), model_path)


if __name__ == "__main__":
    main()

# model = botnet50()
# print_model_named_params(model)