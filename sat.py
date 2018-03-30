import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import SAT_solver
import instance_loader
import itertools

def timestamp():
	return time.strftime( "%Y%m%d%H%M%S", time.gmtime() )
#end timestamp

def memory_usage():
	pid=os.getpid()
	s = next( line for line in open( '/proc/{}/status'.format( pid ) ).read().splitlines() if line.startswith( 'VmSize' ) ).split()
	return "{} {}".format( s[-2], s[-1] )
#end memory_usage

if __name__ == "__main__":
	time_steps = 32
	batch_size = 128
	epochs = 100
	d = 64
	
	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	solver = SAT_solver( d )

	# Create batch loader
	print( "{timestamp}\t{memory}\tLoading instances ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	generator = instance_loader.InstanceLoader( "./instances" )

	# Disallow GPU use
	config = tf.ConfigProto( device_count = {"GPU":0})
	with tf.Session(config=config) as sess:
		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )
		# Run for a number of epochs
		print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
		for b, batch in enumerate( generator.get_batches( batch_size ) ):
			# Build feed_dict
			feed_dict = {
				solver.time_steps: time_steps,
				solver.M: batch.get_sparse_matrix(),
				solver.instance_SAT: np.array( list( 1 if sat else -1 for sat in batch.sat ) ),
				solver.num_vars_on_instance: batch.n
			}
			# Run session
			_, pred_SAT, loss_val, accuracy_val = sess.run(
					[ solver.train_step, solver.predicted_SAT, solver.loss, solver.accuracy ],
					feed_dict = feed_dict
			)
			# Print train step loss and accuracy, as well as predicted sat values compared with the normal ones
			print(
				"{timestamp}\t{memory}\tBatch {batch} Loss: {loss} Accuracy: {accuracy} PRED/REAL: {sat}".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					batch = b,
					loss = loss_val,
					accuracy = accuracy_val,
					sat = list( zip( list( (True if v >= 0.5 else ( False if v <= -0.5 else None ) ) for v in pred_SAT ), batch.sat )  )
				)
			)
		#end for
	#end with
pass
