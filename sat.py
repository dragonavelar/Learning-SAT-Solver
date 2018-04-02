import sys, os, time
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

def run_and_log_batch( epoch, b, batch, time_steps, train = True ):
		sat = list( 1 if sat else 0 for sat in batch.sat )
		# Build feed_dict
		feed_dict = {
			solver.time_steps: time_steps,
			solver.M: batch.get_sparse_matrix(),
			solver.instance_SAT: np.array( sat ),
			solver.num_vars_on_instance: batch.n
		}
		# Run session
		if train:
			_, pred_SAT, loss_val, accuracy_val = sess.run(
					[ solver.train_step, solver.predicted_SAT, solver.loss, solver.accuracy ],
					feed_dict = feed_dict
			)
		else:
			pred_SAT, loss_val, accuracy_val = sess.run(
					[ solver.predicted_SAT, solver.loss, solver.accuracy ],
					feed_dict = feed_dict
			)
		#end if
		# Print train step loss and accuracy, as well as predicted sat values compared with the normal ones
		print(
			"{timestamp}\t{memory}\tEpoch {epoch} Batch {batch} (n,m) ({n},{m}) Loss: {loss} Accuracy: {accuracy} PRED/REAL:".format(
				timestamp = timestamp(),
				memory = memory_usage(),
				epoch = epoch,
				batch = b,
				loss = loss_val,
				accuracy = accuracy_val,
				n = batch.total_n,
				m = batch.total_m
			),
			end = ""
		)
		#for pred, real in list( zip( pred_SAT, sat ) ):
		#	if abs( pred - real ) < 0.5:
		#		print( " [{pred:.2f}, {real:.2f}]".format( pred = pred, real = real ), end = "" )
		#	else:
		#		print( " ({pred:.2f}, {real:.2f})".format( pred = pred, real = real ), end = "" )
		#	#end if
		##end for
		print(flush = True)
#end run_and_log_batch

if __name__ == "__main__":	
	epochs = 100
	d = 128
	
	time_steps = 26
	batch_size = 64
	batches_per_epoch = 50
	# Since test instances are bigger
	test_time_steps = 64
	test_batch_size = 8
	
	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	solver = SAT_solver( d )

	# Create batch loader
	print( "{timestamp}\t{memory}\tLoading instances ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	generator = instance_loader.InstanceLoader( "./instances" )
	test_generator = instance_loader.InstanceLoader( "./test_instances" )

	# Disallow GPU use
	config = tf.ConfigProto( device_count = {"GPU":0})
	with tf.Session(config=config) as sess:
		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )
		# Run for a number of epochs
		print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
		for epoch in range( epochs ):
			print( "{timestamp}\t{memory}\tTRAINING SET".format( timestamp = timestamp(), memory = memory_usage() ) )
			generator.reset()
			for b, batch in itertools.islice( enumerate( generator.get_batches( batch_size ) ), batches_per_epoch ):
				run_and_log_batch( epoch, b, batch, time_steps )
			#end for
			print( "{timestamp}\t{memory}\tTEST SET".format( timestamp = timestamp(), memory = memory_usage() ) )
			test_generator.reset()
			for b, batch in enumerate( test_generator.get_batches( test_batch_size ) ):
				run_and_log_batch( epoch, b, batch, test_time_steps, train = False )
			#end for
		#end for
	#end with
pass
