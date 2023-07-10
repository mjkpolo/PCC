#!/usr/bin/env python

import torch
import argparse
import os
import pickle

class Noema(torch.nn.Module):
    def __init__(self, data,
                 M=20, N=6, B=5,
                 stype=torch.int64,
                 device=torch.device("cpu")):
        super(Noema, self).__init__()
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
        self.wprime = torch.empty((self.Mprime, self.N), dtype=self.stype, device=self.device)
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

def load_data(device, digit, event):
    dir = f'./{device}/{digit}/{event}'
    files = os.listdir(dir)
    N = len(files) # one neuron per channel
    # get tuple array from pickle files in dir in alphabetical order
    data = tuple(pickle.load(open(f'{dir}/{f}', 'rb')) for f in sorted(files))
    B = max(max(d) for d in data)
    M = len(data[0])
    for d in data[1:]:
        assert len(d) == M

    return (N, B, M, data)

def main(args):
    # Small scale test on M1 mac
    stype=torch.int64
    device = torch.device("mps")

    tN, tB, tMprime, tData = load_data(args.device, args.template_digit, args.template_event)
    template_tensor = torch.tensor(tData, dtype=stype, device=device)
    samples = tuple(load_data(args.device, args.sample_digit, sample_event) for sample_event in args.sample_events)
    tL = len(tData)

    sample_tensor = torch.empty((tL, 0), dtype=stype, device=device)
    for s in samples:
        N = max(tN, s[0])
        B = max(tB, s[1])
        Mprime = max(tMprime, s[2])
        assert len(s[3]) == tL
        sample_tensor = torch.cat((sample_tensor, torch.tensor(s[3], dtype=stype, device=device)), dim=1)

    
    # transpose template tensor
    model = Noema(torch.transpose(template_tensor,1,0), stype=stype, device=device, N=N, M=Mprime*B, B=B)

    m = 0.0
    for d in torch.transpose(sample_tensor,1,0):
        r = model(d)
        m = max(r, m)
        print(r)
    print(f"max: {m}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch model of Noema')
    parser.add_argument('--template-digit', '-td', type=int, required=True)
    parser.add_argument('--template-event', '-te', type=int, required=True)
    parser.add_argument('--sample-digit', '-sd', type=int, required=True)
    parser.add_argument('--sample-events', '-se', type=int, nargs='+', required=True)
    parser.add_argument('--device', '-d', type=str, required=True)
    args = parser.parse_args()
    exit(main(args))
