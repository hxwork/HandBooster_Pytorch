import os
from typing import OrderedDict
import torch
import torch.nn as nn
from resnetstack import ResnetStack_Backbone
from resnetstack import Bottleneck


def show_weight(Weight: OrderedDict):
    for k, v in Weight.items():
        # if k.startswith('resnet_stack2'):
        #     break
        print(f'{k: <50}', end='')
        print(v.shape)


def update_weights(pretrained, edited):
    '''
    pretrained: pretrained ResNet50 weights
    edited: edited model weights, model.state_dict()
    return: updated new model weights
    '''
    pretrained_layer_name_size = 60

    for k, v in pretrained.items():
        names = k.split('.')

        if names[0] in ['conv1', 'bn1']:
            # same name
            assert (k in edited)
            edited[k] = v
            print(f'{k: <{pretrained_layer_name_size}}', end='')
            print(k)  # pretrained layer

        elif names[0] in ['layer1', 'layer2', 'layer3', 'layer4']:
            for i in range(1, 3):
                new_layer_name = f'resnet_stack{i}.' + k
                assert (new_layer_name in edited)  # make sure the layer is in new model
                edited[new_layer_name] = v
                print(f'{new_layer_name: <{pretrained_layer_name_size}}', end='')
                print(k)  # pretrained layer

        else:  # no mapped layer
            print(' ' * pretrained_layer_name_size, end='')
            print(k)  # pretrained layer

    return edited


# def check_if_weights_are_updated(pretrained, edited):
#     name = 'resnet_stack1.layer2.2.bn1.bias'
#     origin = edited[name]

#     edited = update_weights(pretrained, edited)
#     edited = edited[name]

#     pretrained = Weight_Resnet['layer2.2.bn1.bias']

#     print(origin)
#     print(edited)
#     print(pretrained)


def generate_pth():
    PATH_RESNET = '/uac/gds/xuhao/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
    assert (os.path.isfile(PATH_RESNET))

    print('torchvision - resnet50 weight found, loading...')
    Weight_Resnet = torch.load(PATH_RESNET)

    # my model
    model = ResnetStack_Backbone(Bottleneck, [3, 4, 6, 3])
    Weight_RNStack = model.state_dict()

    # same_name = ['conv1', 'bn1']
    # stk1_name = ['layer1', 'layer2', 'layer3', 'layer4']
    # stk2_name = ['layer1', 'layer2', 'layer3', 'layer4']

    Weight_RNStack = update_weights(Weight_Resnet, Weight_RNStack)
    torch.save(Weight_RNStack, 'resnetstack.pth')
