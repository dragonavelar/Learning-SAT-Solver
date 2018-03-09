import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_model
import itertools

import generator

PROFILING = 5

def timestamp():

	return time.strftime( "%Y%m%d%H%M%S", time.gmtime() )
#end timestamp

def memory_usage():
	pid=os.getpid()
	s = next( line for line in open( '/proc/{}/status'.format( pid ) ).read().splitlines() if line.startswith( 'VmSize' ) ).split()
	return "{} {}".format( s[-2], s[-1] )
#end memory_usage

if __name__ == "__main__":
	time_steps = 12
	batch_size = 32
	epochs = 20
	n = 5
	m = 50
	d = 32
	Lmsg_sizes 	= [2*n*d,	2*n*d,	2*n*d]
	Cmsg_sizes 	= [m*d, 	m*d,	m*d]
	Lvote_sizes = [32,		32,		32]
	
	# Build model
	print("Building model ...")
	M, pred_SAT, label_SAT, loss, train_step, var_dict = build_model( 
		time_steps = time_steps,
		batch_size = batch_size,
		d = d,
		n = n,
		m = m,
		Lmsg_sizes = Lmsg_sizes,
		Lvote_sizes = Lvote_sizes,
		Cmsg_sizes = Cmsg_sizes
)

	# Create batch generator
	print("Creating batch generator ...")
	generator = generator.generate(n, m, batch_size=batch_size)


	if PROFILING:
		print( "Setting up profiling ..." )
		builder = tf.profiler.ProfileOptionBuilder
		opts_op = builder( builder.time_and_memory() )
		opts_op.with_file_output( "./profiling/op.out" )	
		opts_op = opts_op.build()
		
		opts_scope = builder( builder.trainable_variables_parameter() )
		opts_scope.with_file_output( "./profiling/scope.out" )
		opts_scope = opts_scope.build()
		
		opts_graph = builder()
		opts_graph.with_timeline_output( "./profiling/graph.timeline" )
		opts_graph = opts_graph.build()
		
		opts_code = builder()
		opts_code.with_timeline_output( "./profiling/code.timeline" )
		opts_code.with_pprof_output( "./profiling/code.pprof" )
		opts_code = opts_code.build()
		
		with tf.contrib.tfprof.ProfileContext('./profiling/profilecontext',
			trace_steps = range( 0, epochs ),
			dump_steps = [ epochs - 1 ]
		) as pctx:
			pctx.add_auto_profiling( 'op', opts_op, range( 0, epochs, PROFILING ) )
			pctx.add_auto_profiling( 'scope', opts_scope, range( 0, epochs, PROFILING ) )
			pctx.add_auto_profiling( 'graph', opts_graph, range( 0, epochs, PROFILING ) )
			pctx.add_auto_profiling( 'code', opts_code, range( 0, epochs, PROFILING ) )
			
			# Allow GPU memory growth
			config = tf.ConfigProto()
			config.gpu_options.per_process_gpu_memory_fraction = 0.9
			
			with tf.Session(config=config) as sess:
				# Initialize global variables
				print("Initializing global variables ... ")
				sess.run( tf.global_variables_initializer() )
				# Run for a number of epochs
				print("Running for {} epochs".format(epochs))
				epoch = 1
				for epoch, batch in zip( range(epochs), generator ):
					# Get features, labels
					features, labels = batch
					# Run session
					_, _, loss_val = sess.run(
						[train_step, pred_SAT, loss],
						feed_dict = {
							M: features,
							label_SAT: labels
						}#,
						#options = tf.RunOptions(
						#	trace_level = tf.RunOptions.FULL_TRACE
						#),
						#run_metadata = run_metadata
					)
					# Print train step and loss
					print(
						"{timestamp}\t{memory}\tEpoch {epoch} Loss: {loss}".format(
							timestamp = timestamp(),
							memory = memory_usage(),
							epoch = epoch,
							loss = loss_val
						)
					)
				#end for
			#end with
	else:
		# Allow GPU memory growth
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			# Initialize global variables
			print("Initializing global variables ... ")
			sess.run( tf.global_variables_initializer() )
			# Run for a number of epochs
			print("Running for {} epochs".format(epochs))
			epoch = 1
			for epoch, batch in zip( range(epochs), generator ):
				# Get features, labels
				features, labels = batch
				# Run session
				_, _, loss_val = sess.run( [train_step, pred_SAT, loss], feed_dict={M: features, label_SAT: labels} )
				# Print train step and loss
				print(
					"{timestamp}\t{memory}\tEpoch {epoch} Loss: {loss}".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						loss = loss_val
					)
				)
			#end for
		#end with

pass
