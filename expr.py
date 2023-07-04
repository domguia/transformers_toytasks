#!/usr/bin/env python

import math

import torch, torchvision

from torch import nn
from torch.nn import functional as F

def random_var(nb_variables=None, variables=None):
    if variables is None:
        return chr(ord('A') + torch.randint(nb_variables, (1,)).item())
    else:
        l = list(variables)
        return l[torch.randint(len(l), (1,)).item()]

def random_expr(variables, budget):
    if budget <= 5:
        op=torch.randint(2, (1,)).item()
        if op == 0 and len(variables) > 0:
            return random_var(variables=variables)
        else:
            return str(torch.randint(10, (1,)).item())
    else:
        op=torch.randint(4, (1,)).item()
        if op == 0:
            e=random_expr(variables,budget-2)
            if ("+" in e or "-" in e or "*" in e) and (e[0]!="(" or e[-1]!=")"):
                return "("+e+")"
            else:
                return e
        else:
            b = 2 + torch.randint(budget-5, (1,)).item()
            e1=random_expr(variables,b)
            e2=random_expr(variables,budget-b-1)
            if op == 1:
                return e1+"+"+e2
            elif op == 2:
                return e1+"+"+e2
            elif op == 3:
                return e1+"*"+e2

def generate_program(nb_variables, length):
    s = ""
    variables = set()
    while len(s) < length:
        v = random_var(nb_variables=nb_variables)
        s += v+"="+random_expr(variables,budget = min(20,length-3-len(s)))+";"
        variables.add(v)
    return s, variables

def generate_sequences(nb, nb_variables = 5, length=20):
    sequences=[]
    for n in range(nb):
        result = None
        while result==None or max(result.values())>100:
            p,v=generate_program(nb_variables, length)
            v=", ".join([ "\""+v+"\": "+v for v in v ])
            ldict={}
            exec(p+"result={"+v+"}",globals(),ldict)
            result=ldict["result"]

        k=list(result.keys())
        k.sort()
        sequences.append(p+" "+";".join([v+":"+str(result[v]) for v in k]))

    return sequences

if __name__ == "__main__":
    import time
    start_time = time.perf_counter()
    sequences=generate_sequences(1000)
    end_time = time.perf_counter()
    for s in sequences[:10]:
        print(s)
    print(f"{len(sequences) / (end_time - start_time):.02f} samples per second")
