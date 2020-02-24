"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import os
import csv
import numpy
from sklearn import preprocessing
import urllib
import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

import matplotlib.pyplot as plt
import networkx as nx
#import pandas
import scipy
import numpy
from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf

def degre_tf(tensor_obj):
    tensor_obj=tf.reshape(tensor_obj,(5,35,35))
    shapes = tensor_obj.get_shape().as_list()
    pivot_np = np.zeros((shapes[0], shapes[1]))
    deg = tf.Variable(pivot_np)
    for i in range(shapes[0]):#  testing for one sample
        G = nx.Graph()
        graph_tensor = tensor_obj[i]
        for x in range(shapes[1]):
            G.add_node(x) ##  adding 35 nodes in the graph
        for x in range(shapes[1]):  # or:  for x in range(shapes[1])
            for y in range(shapes[2]):
                  my_tensor=graph_tensor[x][y]
                  print('my tensor is',my_tensor)
                  #gt=tf.Variable(graph_tensor[x][y],dtype=tf.float64)
                  if tf.math.not_equal(graph_tensor[x][y], 0):
                    G.add_edge(x,y,weight=graph_tensor[x][y])

        dict = nx.degree_centrality(G)
        #dict=nx.closeness_centrality(G,distance='weight',wf_improved=False)
        #dict=nx.eigenvector_centrality_numpy(G, weight='weight')
        #dict=nx.betweenness_centrality(G, weight= 'weight', endpoints=False,normalized=True)# default settings
        for z in range(shapes[1]):
            pivot_np[i][z] = dict.get(z)

    print('degrees are', pivot_np)
    deg.assign(pivot_np)
    tf.cast(deg, tf.float32)
    return deg
def load_data(path): #,type_,size,dataset):



    reg=np.load('vtrain1.npy')# train file for the first graph (ground truth) (orginal graph) input data
    mal=np.load('vtrain2.npy') # your output (generated graph (tranlating to a new graph)

    data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
    for i in range(reg.shape[0]):
       data[i,:,:,0]=reg[i]
       data[i,:,:,1]=mal[i]

       return data
def load_data_test(size,dataset):

    reg=np.load('vtest1.npy')
    data=np.zeros((reg.shape[0],reg.shape[1],reg.shape[2],2))
    for i in range(reg.shape[0]):
      data[i,:,:,0]=reg[i]
      data[i,:,:,1]=reg[i]
    return data


def deg(reala,sesss): ##  altered code

    d = []

    #print('burasi'+str(type(selff)))
    view2 = reala.eval(session=sesss )
    for i in range (0,len(view2)):
        matrixx = np.asmatrix(view2[i])
        matrixx = np.where(matrixx > 0, 1, 0)
        d[i] = matrixx
    return d

import networkx as nx
from collections import defaultdict
def bc(G):
    vertices = G.nodes()
    new_bc = {}
    paths = defaultdict(dict)

    # Get shortest paths between all pairs of vertices
    for i, vertex in enumerate(list(vertices)[:-1]):
        for o_vertex in list(vertices)[i+1:]:
            paths[vertex][o_vertex] = [path for path in
                                       nx.all_shortest_paths(G, vertex, o_vertex)]

    for vertex in vertices:
        counter = 0
        for i, vertex1 in enumerate(list(vertices)[:-1]):
            for vertex2 in list(vertices)[i+1:]:
                for path in paths[vertex1][vertex2]:
                    if vertex in path[1:-1]:
                        counter += 1
        new_bc[vertex] = counter

    return new_bc
'''
def degre_tf2(tensor_obj):
    deg=tf.reduce_sum(tensor_obj,1)
    print('deg is shape',deg)
    return deg
'''
def degre_tf(tensor_obj):
    tf.enable_eager_execution()
    tensor_obj=tf.reshape(tensor_obj,(30,35,35))
    shapes = tensor_obj.get_shape().as_list()
    pivot_np = np.zeros((shapes[0], shapes[1]))
    deg = tf.Variable(pivot_np)
    for i in range(shapes[0]):#  testing for one sample
        #G = nx.Graph()
        graph_tensor = tensor_obj[i]
        #graph_tensor = tf.Variable(graph_tensor, dtype=tf.float64)
        graph_tensor=graph_tensor.numpy()
        G = nx.from_numpy_matrix(graph_tensor)
        '''
        for x in range(shapes[1]):
            G.add_node(x) ##  adding 35 nodes in the graph
        for x in range(shapes[1]):  # or:  for x in range(shapes[1])
            for y in range(shapes[2]):
                  my_tensor=graph_tensor[x][y]
                  print('my tensor is',my_tensor)
                  #gt=tf.Variable(graph_tensor[x][y],dtype=tf.float64)
                  if tf.math.not_equal(graph_tensor[x][y], 0):
                    G.add_edge(x,y,weight=graph_tensor[x][y])
        '''
        dict = nx.degree_centrality(G)
        #dict=nx.closeness_centrality(G,distance='weight',wf_improved=False)
        #dict=nx.eigenvector_centrality_numpy(G, weight='weight')
        #dict=nx.betweenness_centrality(G, weight= 'weight', endpoints=False,normalized=True)# default settings
        for z in range(shapes[1]):
            pivot_np[i][z] = dict.get(z)

    print('degrees are', pivot_np)
    deg.assign(pivot_np)
    tf.cast(deg, tf.float32)
    return deg
