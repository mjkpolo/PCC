#!/usr/bin/env python

import torch
import argparse
import os


def main(args):
    stype=torch.int64
    device = torch.device('mps')

    N = args.neurons
    B = args.batch
    Mprime = args.m_prime

    torch.manual_seed(args.seed)
    tensor = torch.randint(0, B, (Mprime, N), dtype=stype, device=device)
    if args.join is not None:
        with open(args.join, 'rb') as f:
            tensor = torch.cat((torch.load(f, map_location=torch.device('mps')), tensor), dim=0)
    with open(args.output, 'wb') as f:
        torch.save(tensor, f)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch model of PCC')
    parser.add_argument('-n', '--neurons', help='Number of neurons', type=int, required=True)
    parser.add_argument('-p', '--m-prime', help='batches per sample', type=int, required=True)
    parser.add_argument('-b', '--batch', help='Unary bits per batch', type=int, required=True)
    parser.add_argument('-s', '--seed', help='Random seed', type=int, required=True)
    parser.add_argument('-o', '--output', help='Output of tensor', type=str, required=True)
    parser.add_argument('-j', '--join', help='Tensor to append to', type=str)
    args = parser.parse_args()
    exit(main(args))
