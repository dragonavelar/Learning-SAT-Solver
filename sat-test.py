import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import SAT_solver
import instance_loader
import itertools
from util import test_with, timestamp, memory_usage

if __name__ == "__main__":
	d = 128
	batch_size = 64
	test_time_steps = 28 # Use a much bigger number of time steps
	test_batch_size = batch_size
	
	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	solver = SAT_solver( d )


	# Create model saver
	saver = tf.train.Saver()

	with tf.Session(config=config) as sess:

		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )
		
		# Restore saved weights
		print( "{timestamp}\t{memory}\tRestoring saved model ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		saver.restore(sess, "./tmp/model.ckpt")

		# Test SR distribution
		test_with(
			sess,
			solver,
			"./test_instances",
			"SR",
			time_steps = test_time_steps
		)
		# Test Phase Transition distribution
		test_with(
			sess,
			solver,
			"./critical_instances_40",
			"PT40",
			time_steps = test_time_steps
		)
		test_with(
			sess,
			solver,
			"./critical_instances_80",
			"PT80",
			time_steps = test_time_steps
		)
pass
