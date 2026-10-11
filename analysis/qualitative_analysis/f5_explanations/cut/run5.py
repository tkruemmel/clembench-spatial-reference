import collections, json, pickle
from order import *
W=[]; A=[]
for rec in episodes():
    ins, ev = load(rec)
    for w in waiting(rec, ins, ev):
        w["rec"]=rec; W.append(w)
    for a in arrivals(rec, ins, ev):
        a["rec"]=rec; A.append(a)
pickle.dump((W,A),open("dyn.pkl","wb"))
