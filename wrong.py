# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 20:40:35 2018

@author: gxjco
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import scipy.misc
import numpy as np
from model import graph2graph
#import tensorflow as tf
import datetime
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
#from numba import cuda
import gc
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
#parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=30, help='# graphs in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default='200', help='# graphs used to train')#154
parser.add_argument('--ngf', dest='ngf', type=int, default=5, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=5, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output channels')
parser.add_argument('--niter', dest='niter', type=int, default=10, help='# of iter at starting learning rate')
parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.0005  , help='initial learning rate for adam')
parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.00009  , help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the graphs for data argumentation')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=1, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=10, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes graphsin order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial graph list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./mymodel', help='models are saved here,need to be distinguishable for different dataset')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='.vtest1', help='test sample are saved here, need to be distinguishable for different dataset')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--train_dir', dest='train_dir', default='./', help='train sample are saved here')
parser.add_argument('--graph_size', dest='graph_size', default=[35,35], help='size of graph')
parser.add_argument('--output_size', dest='output_size', default=[35,35], help='size of graph')
parser.add_argument('--dataset', dest='dataset', default='authentication', help='chose from authentication, scale-free and poisson-random')
args = parser.parse_args()
tf.executing_eagerly()
#tf.reset_default_graph()
tf.compat.v1.reset_default_graph
def main():
    start = datetime.datetime.now()
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    
    
    
     
    #with tf.Session() as sess:
        #model = graph2graph(sess, batch_size=args.batch_size,checkpoint_dir=args.checkpoint_dir,sample_dir=args.sample_dir,test_dir=args.test_dir,train_dir=args.train_dir,graph_size=args.graph_size,output_size=args.output_size, dataset=args.dataset)
    v1=np.load('data1s.npy')
    v4=np.load('data2s.npy')

    predicted = np.zeros(shape=(30,1225))#  full predicted  dataset
    ground = np.zeros(shape=(30,1225)) #  structure for test data (unused for traınıng) ın each ıteratıon
    predicted2 = np.zeros(shape=(30,1225))#  full predicted  dataset
    ground2 = np.zeros(shape=(30,1225)) #  structure for test data (unused for traınıng) ın each ıteratıon
    #loo = KFold(n_splits=5)
    #loo.get_n_splits(v1)

    #KFold(n_splits=6, random_state=None, shuffle=False)
    from sklearn.model_selection import KFold
    loo=KFold(n_splits=5)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    option=None

    #TF_CONFIG = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5),allow_soft_placement=True)

    #sess = tf.Session(config=TF_CONFIG)
   
    pcr=[]
    mser=[]
    pv=[]
    predicteddata=[]
    wholepredicted=[]
    pcr2=[]
    mser2=[]
    pv2=[]
    predicteddata2=[]
    wholepredicted2=[]
    for train_index, test_index in loo.split(v1):
        tf.compat.v1.reset_default_graph()
        tf.enable_eager_execution()
        with tf.compat.v1.Session() as sess:
            tf.enable_eager_execution()
            model = graph2graph(sess, batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir,test_dir=args.test_dir,train_dir=args.train_dir,graph_size=args.graph_size,output_size=args.output_size, dataset=args.dataset)

            #epoch=args.epoch
            ##cuda.select_device(0)

            reg1, reg2 = v1[train_index], v1[test_index]#  reg1 ıs v1 traın dataset, reg2 v1 testdataset
            print(np.shape(reg1),'reg1')
            print(np.shape(reg2),'reg2')
            mal1, mal2 = v4[train_index], v4[test_index]#mal1:v4 traın dataset,mal2 v4 testdataset(lıke y traın and test

            datatrain=np.zeros((reg1.shape[0],reg1.shape[1],reg1.shape[2],2)) #  forming a paır of traın samples
            print('now the test index is', test_index)
            for i in range(reg1.shape[0]):
                datatrain[i,:,:,0]=reg1[i]  ###reg1=train v1  reg ıs A   and mal ıs B
                datatrain[i,:,:,1]=mal1[i]  #mal1=train v2 (154, 35, 35, 2)
            print(np.shape(datatrain), 'thıs ıs traın data after loop')
            model.train(args, datatrain)



            datatest=np.zeros((reg2.shape[0],reg2.shape[1],reg2.shape[2],2))  #  here we dont use any v4 data
            for i in range(reg2.shape[0]):
                datatest[i,:,:,0]=reg2[i]
                datatest[i,:,:,1]=reg2[i]#  forming a paır of test samples all from v1.



            print(predicted.shape)
            xx=model.test(args,   datatest)
            predicted = xx.reshape(-1,1225)  #   predicted[test_index,:] fill predicted with current test_index
            ground = np.reshape(mal2, (-1,1225)) #  fill the previously defined predicted array with current test data
            print(predicted.shape,ground.shape)
            corr,pvalue = pearsonr(predicted.ravel(),ground.ravel())
            mse=mean_squared_error(predicted,ground)
            pcr.append(corr)
            pv.append(pvalue)
            mser.append(mse)
            predicteddata.append(predicted)
            wholepredicted.append(predicted)
            np.savetxt("./pred/pred"+str(test_index[0])+".txt", predicted, delimiter=',')
            np.savetxt("./gt/gt"+str(test_index[0])+".txt", ground, delimiter=',')


            print("pearson corr ", corr)
            print("pvalue ", pvalue)
            print("mean square error ",mse)
            print('###### begin B to A##################################################################################')
            datatest2=np.zeros((mal2.shape[0],mal2.shape[1],mal2.shape[2],2))  #  here we dont use any v4 data
            for i in range(reg2.shape[0]):
                datatest2[i,:,:,0]=mal2[i]
                datatest2[i,:,:,1]=mal2[i]#  forming a paır of test samples all from v1.



            print(predicted2.shape)
            xx=model.test2(args,   datatest2)
            predicted2 = xx.reshape(-1,1225)  #   predicted[test_index,:] fill predicted with current test_index
            ground2 = np.reshape(reg2, (-1,1225)) #  fill the previously defined predicted array with current test data

            sess.close()
            gc.collect()
            print(predicted2.shape,ground2.shape)
            corr2,pvalue2 = pearsonr(predicted2.ravel(),ground2.ravel())
            mse2=mean_squared_error(predicted2,ground2)
            pcr2.append(corr2)
            pv2.append(pvalue2)
            mser2.append(mse2)
            predicteddata2.append(predicted2)
            wholepredicted2.append(predicted2)
            np.savetxt("./pred/pred2"+str(test_index[0])+".txt", predicted2, delimiter=',')
            np.savetxt("./gt/gt2"+str(test_index[0])+".txt", ground2, delimiter=',')


            print("pearson corr ", corr2)
            print("pvalue ", pvalue2)
            print("mean square error ",mse2)



            print('###### end of inner B to A#############################################################################')
            end = datetime.datetime.now()
            print (start, 'thıs ıs the start tıme')
            print (end, 'thıs ıs the end  tıme')
            print ('total tıme ıs', start-end)
            gc.collect()

        tf.reset_default_graph()



    with open("./savedrestc/pcr rhv4 to v3last.txt", "w") as f:
        for s in pcr:
            f.write(str(s) +"\n")
    with open("./savedrestc/mser rhv4 to v3last.txt", "w") as f:
        for p in mser:
            f.write(str(p) +"\n")
    with open("./savedrestc/pv rhv4 to v3last.txt", "w") as f:
        for v in pv:
            f.write(str(v) +"\n")
    with open("./predwhole/predwhole.txt", "w") as f:
        for w in predicteddata:
            f.write(str(w) +"\n")

    #pv_mean=pv/5
    pcr_mean=sum(pcr) / len(pcr)
    print("length of pv is ", len(pv))
    print("length of pcr is ", len(pcr))
    mse_mean=sum(mser) / len(mser)
    pv_mean =sum(pv) / len(pv)
    print("****Now mean results A to B**** ")
    print("pearson corr_mean ", pcr_mean)
    print("mean square error_mean ",mse_mean)
    print("pv_mean is ",pv_mean)
    end = datetime.datetime.now()
    print (start, 'thıs ıs the start tıme')
    print (end, 'thıs ıs the end  tıme')
    print ('total tıme ıs', start-end)
    #np.savetxt("./predwhole/predicted3d.txt.txt",predicteddata)
    predicteddata2d=predicteddata[1:2]
    predicted2d=np.reshape(predicteddata2d,(30,1225))
    np.savetxt("./predwhole/2dpredictedatob.txt",predicted2d)
    np.save("./predwhole/wholepredicteddatatob.npy", wholepredicted)
    print('################## beginning of second part B to A #############################################')
    pcr_mean2=sum(pcr2) / len(pcr2)
    mse_mean2=sum(mser2) / len(mser2)
    pv_mean2 =sum(pv2) / len(pv2)
    print("****Now mean results for B to A **** ")
    print("pearson corr_mean_B to A ", pcr_mean2)
    print("mean square error_mean_B to A ",mse_mean2)
    print("pv_mean is ",pv_mean2)
    predicteddata2d2=predicteddata2[1:2]
    predicted2d2=np.reshape(predicteddata2d2,(30,1225))
    np.savetxt("./predwhole/2dpredictedbtoa.txt",predicted2d2)
    np.save("./predwhole/wholepredictedbtoa.npy", wholepredicted2)
    print('################## end of second part B to A #############################################')
    end = datetime.datetime.now()
    print (start, 'thıs ıs the start tıme')
    print (end, 'thıs ıs the end  tıme')
    print ('total tıme ıs', start-end)
if __name__ == '__main__':
    
   
    main()
      #tf.app.run()
    
#from memory_profiler import memory_usage
 #   m=memory_usage(main(), interval=20, timeout=240)
  #  print(max(m))'''

