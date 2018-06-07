import torch.nn
import torch
from segment import *
from drn import BasicBlock
import data_transforms

def calculate_input_size(model, target_output=(20,20)):

    params = []
    for c in model.children():
        helper(c, params)

    in_size = ()
    for o in target_output:
        in_size += (o,)
    for p in reversed(params):
        in_size = input_size(in_size, p[0], p[1], p[2])
    return in_size

def helper(layer, params):
    if isinstance(layer, torch.nn.Conv2d):
        params.append((layer.kernel_size, layer.dilation, layer.stride))
        layer.padding = (0,0)
    elif isinstance(layer, torch.nn.Sequential):
        for element in layer:
            helper(element, params)
    elif isinstance(layer, BasicBlock):
        helper(layer.conv1, params)
        helper(layer.conv2, params)
        if layer.residual:
            pass
        # layer.residual = False

def input_size(output_size ,kernel, dilation, stride):
    input = ()
    for i in range(len(output_size)):
        input += (output_size[i]*stride[i] + dilation[i]*(kernel[i] - 1),)
    return input

def main():
    model = DRNSeg('drn_d_22', 11)
    target_output = (160, 160)
    input_size = calculate_input_size(model)
    print(input_size, 'calculated input shape')

    input = torch.randn(4, 3, input_size[0], input_size[1])
    # input_var = torch.autograd.Variable(input)
    # output = model(input_var)[0]
    # print(output.shape,'output shape') #torch.Size([4, 11, 640, 480])
    # print(target_output, 'target output')
    # assert output.shape[-2:] == target_output
    print(data_transforms.center_crop(input, 2,2).size())

if __name__ == "__main__":
    main()