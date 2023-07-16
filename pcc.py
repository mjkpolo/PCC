#!/usr/bin/env python

import torch
import argparse
import os
import pickle

class PCC(torch.nn.Module):
    def __init__(self, data,
                 M=20, N=6, B=5,
                 stype=torch.int64,
                 device=torch.device("mps")):
        super(PCC, self).__init__()
        self.timestep = 0
        self.stype = stype
        self.device = device
        self.N = N
        self.M = M
        self.B = B
        self.Mprime = self.M//self.B

        assert self.M % self.B == 0
        assert data.shape == (self.Mprime, self.N)

        self.template = data
        self.wprime = torch.zeros((self.Mprime, self.N), dtype=self.stype, device=self.device)
        self.C1 = torch.tensor(self.Mprime * self.N, dtype=self.stype, device=self.device)
        self.C2 = torch.sum(self.template, dtype=self.stype)
        self.C3 = self.C1*torch.sum(self.template**2, dtype=self.stype)-self.C2**2

    def forward(self, bin):
        assert bin.dtype == self.stype

        if self.timestep < self.Mprime:
            self.wprime[self.timestep] = bin
            self.timestep += 1
            r = torch.tensor(0, dtype=self.stype, device=self.device)
        else:
            self.wprime = torch.cat((self.wprime[1:], bin.unsqueeze(0)), dim=0)
        S1 = torch.sum(self.wprime*self.template, dtype=self.stype)
        S2 = torch.sum(self.wprime, dtype=self.stype)
        S3 = torch.sum(self.wprime**2, dtype=self.stype)
        r = ((self.C1*S1 - self.C2*S2)**2)/(self.C3*(self.C1*S3 - S2**2))
        
        return (r**(1/2))


def main(args):
    # Small scale test on M1 mac
    stype=torch.int64
    device = torch.device("mps")

    N = args.neurons
    B = args.batch
    Mprime = args.m_prime

    template_tensor = torch.load(args.template_data, map_location=device)
    sample_tensor = torch.load(args.sample_data, map_location=device)
    model = PCC(template_tensor, stype=stype, device=device, N=N, M=Mprime*B, B=B)

    d = None
    for d in sample_tensor:
        r = model(d).item()
        print(r)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch model of PCC')
    parser.add_argument('-n', '--neurons', help='Number of neurons', type=int, required=True)
    parser.add_argument('-p', '--m-prime', help='batches per sample', type=int, required=True)
    parser.add_argument('-b', '--batch', help='Unary bits per batch', type=int, required=True)
    parser.add_argument('-t', '--template-data', help='Pickle of template', type=str, required=True)
    parser.add_argument('-s', '--sample-data', help='Pickle of sample', type=str, required=True)
    args = parser.parse_args()
    exit(main(args))
