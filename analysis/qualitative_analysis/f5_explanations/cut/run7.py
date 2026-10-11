import collections, pickle
from order import *
from msgs import tag_episode
import os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
rows=[r for r in pickle.load(open(f"s3rows.{ARM}.pkl","rb")) if r["kind"]=="door"]
TAGS=["a_door_cell","a2_corr_word","b_v_route_door","b1_v_route_doorcell","c0_order_any_v","c_order_hv","d_v_target","d_plan","h_goal_plan"]
out=[]
for r in rows:
    ins,ev=load(r["rec"])
    r["hits"]=tag_episode(r["rec"],ins,ev,r["h"],r["v"],r["win_end"])
pickle.dump(rows,open(f"s3tagged.{ARM}.pkl","wb"))
for pr in PROTOS:
    for o in ("v first","h first"):
        rs=[r for r in rows if r["rec"]["proto"]==pr and r["order"]==o]
        print("%-10s %-8s n=%2d "%(pr,o,len(rs))+" ".join("%s:%d"%(t.split('_')[0],sum(1 for r in rs if r["hits"].get(t))) for t in TAGS))
