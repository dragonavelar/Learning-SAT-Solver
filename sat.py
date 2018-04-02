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
			"{timestamp}\t{memory}\tEpoch {epoch} Batch {batch} (n,m) ({n},{m}) Loss: {loss} Accuracy: {accuracy}".format(
				timestamp = timestamp(),
				memory = memory_usage(),
				epoch = epoch,
				batch = b,
				loss = loss_val,
				accuracy = accuracy_val,
				n = batch.total_n,
				m = batch.total_m
			),
			flush = True
		)
		return loss_val, accuracy_val
#end run_and_log_batch

if __name__ == "__main__":	
	epochs = 1000
	d = 128
	
	time_steps = 26
	batch_size = 64
	batches_per_epoch = 128
	test_time_steps = time_steps
	test_batch_size = batch_size
	
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
			# Reset training generator and run with a sample of the training instances
			print( "{timestamp}\t{memory}\tTRAINING SET BEGIN".format( timestamp = timestamp(), memory = memory_usage() ) )
			generator.reset()
			epoch_loss = 0.0
			epoch_accuracy = 0.0
			for b, batch in itertools.islice( enumerate( generator.get_batches( batch_size ) ), batches_per_epoch ):
				l, a = run_and_log_batch( epoch, b, batch, time_steps )
				epoch_loss += l
				epoch_accuracy += a
			#end for
			epoch_loss = epoch_loss / batches_per_epoch
			epoch_accuracy = epoch_accuracy / batches_per_epoch
			print( "{timestamp}\t{memory}\tTRAINING SET END Mean loss: {loss} Mean Accuracy = {accuracy}".format(
				loss = epoch_loss,
				accuracy = epoch_accuracy,
				timestamp = timestamp(),
				memory = memory_usage()
				)
			)
			# Summarize results and print epoch summary
			test_loss = 0.0
			test_accuracy = 0.0
			test_batches = 0
			# Reset test generator and run with the test instances
			print( "{timestamp}\t{memory}\tTEST SET BEGIN".format( timestamp = timestamp(), memory = memory_usage() ) )
			test_generator.reset()
			for b, batch in enumerate( test_generator.get_batches( test_batch_size ) ):
				l, a = run_and_log_batch( epoch, b, batch, test_time_steps, train = False )
				test_loss += l
				test_accuracy += a
				test_batches += 1
			#end for
			# Summarize results and print test summary
			test_loss = test_loss / test_batches
			test_accuracy = test_accuracy / test_batches
			print( "{timestamp}\t{memory}\tTEST SET END Mean loss: {loss} Mean Accuracy = {accuracy}".format(
				loss = test_loss,
				accuracy = test_accuracy,
				timestamp = timestamp(),
				memory = memory_usage()
				)
			)
		#end for
	#end with
pass
