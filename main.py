import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_model
import itertools

import generator

def timestamp():

	return time.strftime( "%Y%m%d%H%M%S", time.gmtime() )
#end timestamp

def memory_usage():
	pid=os.getpid()
	s = next( line for line in open( '/proc/{}/status'.format( pid ) ).read().splitlines() if line.startswith( 'VmSize' ) ).split()
	return "{} {}".format( s[-2], s[-1] )
#end memory_usage

if __name__ == "__main__":
	time_steps = 26
	batch_size = 32
	epochs = 1000
	n = 40
	m = 400
	d = 128
	Lmsg_sizes 	= [2*n*d]
	Cmsg_sizes 	= [m*d]
	Lvote_sizes = [32]
	
	
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
	#L_vote = var_dict["L_vote"]
	## Run the program
	#with tf.Session() as sess:
	#	sess.run( tf.global_variables_initializer() )
	#	matrix = np.random.rand( batch_size,2*n,m ) < 0.5
	#	satisfiability = np.random.rand( batch_size, ) < 0.5
	#	print( satisfiability )
	#	print( '\n\n' )
	#	for name, thing in zip(
	#		["train_step","pred_SAT","loss"] + [ "L_vote[t={}]".format(i) for i,lv in enumerate( L_vote ) ],
	#		sess.run(
	#			[train_step,pred_SAT,loss] + L_vote,
	#			feed_dict = {
	#				M: matrix.astype( np.float32 ),
	#				label_SAT: ( satisfiability.astype( np.float32 ) - 0.5 ) * 2
	#			}
	#		)
	#	):
	#		print( "{}: {}".format( name, thing ) )
	#	#end for
	##end session

	# Create batch generator
	generator = generator.generate(n, m, batch_size=batch_size)

	with tf.Session() as sess:
		# Initialize everything
		sess.run( tf.global_variables_initializer() )
		# Run for a number of epochs
		for (epoch,batch) in enumerate(itertools.islice(generator,epochs)):
			# Get features, labels
			features, labels = batch
			# Run session
			_, _, loss_val = sess.run( [train_step, pred_SAT, loss], feed_dict={M: features, label_SAT: labels} )
			# Print train step and loss
			print("Loss: {}".format(loss_val))
		#end for
	#end with

pass
