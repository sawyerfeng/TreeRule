from rdflib import Graph, URIRef, Literal, Namespace
from dataset import *
import os
data = "/home/fxy/thesis/treeRule/datasets/family/"
sparsity = 1
inv = True
# 从data字符串中提取出最后一个/ 分割的字符串
data_name = data.split('/')[-2]
print(data_name)
dataset = Dataset(data,sparsity=sparsity,inv=inv)
if data_name+".rdf" not in os.listdir(data):
    
    fact, train, valid, test = dataset.rdf_data_
    train_data = fact + train
    g = Graph()

    ns = Namespace("http://"+data_name+".org/")
    # Add triples to the graph
    for h,r,t in train_data:
        g.add((URIRef(ns+h), URIRef(ns+r), URIRef(ns+t)))
    # 转为.rdf文件
    g.serialize(destination=data+data_name+'.rdf', format='n3')
def query():
    query = """
    SELECT ?x1 ?y1 ?x2 ?y2 ?z
    WHERE {
        # 路径1: h -> R1 -> x1 -> R2 -> y1
        ?h <R1> ?x1 .
        ?x1 <R2> ?y1 .

        # 路径2: h -> R3 -> x2 -> R4 -> y2
        ?h <R3> ?x2 .
        ?x2 <R4> ?y2 .

        # 路径3: h -> R5 -> z
        ?h <R5> ?z .
    }
    """
    query = query.replace("<R1>", str(R1)).replace("<R2>", str(R2)).replace("<R3>", str(R3)).replace("<R4>", str(R4)).replace("<R5>", str(R5))

