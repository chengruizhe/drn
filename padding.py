import torch.nn
import torch
from segment import DRNSeg
from drn import BasicBlock

def calculate_padding(model, target_output=(20,20)):

    params = []
    for c in model.children():
        helper(c, params)

    output_size = ()
    for o in target_output:
        output_size += (o,)
    for p in reversed(params):
        output_size = input_size(output_size, p[0], p[1], p[2])
    return output_size

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
        layer.residual = False

def input_size(output_size ,kernel, dilation, stride):
    input = ()
    for i in range(len(output_size)):
        input += (output_size[i]*stride[i] + dilation[i]*(kernel[i] - 1),)
    return input

def main():
    model = DRNSeg('drn_d_22', 11)
    target_output = (160, 160)
    input_size = calculate_padding(model)
    print(input_size, 'calculated input shape')

    input = torch.randn(4, 3, input_size[0], input_size[1])
    input_var = torch.autograd.Variable(input)
    output = model(input_var)[0]
    print(output.shape,'output shape') #torch.Size([4, 11, 640, 480])
    print(target_output, 'target output')
    assert output.shape[-2:] == target_output

if __name__ == "__main__":
    main()