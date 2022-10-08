import torch
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--input', default='', type=str)
    parser.add_argument('--output', default='', type=str)
    return parser

if __name__=='__main__':
    parser = argparse.ArgumentParser('model convert', parents=[get_args_parser()])
    args = parser.parse_args()

    model = torch.load(args.input)
    if len(args.output) > 0:
        torch.save(model['model'], args.output, _use_new_zipfile_serialization=False)
    else:
        torch.save(model['model'], args.input, _use_new_zipfile_serialization=False)
