import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_model

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
	batch_size = 3
	n = 5
	m = 7
	d = 11
	Lmsg_sizes = [1,2,3]
	Lvote_sizes = [1,2,3]
	Cmsg_sizes = [1,2,3]
	
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
	L_vote = var_dict["L_vote"]
	# Run the program
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		matrix = np.random.rand( batch_size,2*n,m ) < 0.5
		satisfiability = np.random.rand( batch_size, ) < 0.5
		print( satisfiability )
		print( '\n\n' )
		for name, thing in zip(
			["train_step","pred_SAT","loss"] + [ "L_vote[t={}]".format(i) for i,lv in enumerate( L_vote ) ],
			sess.run(
				[train_step,pred_SAT,loss] + L_vote,
				feed_dict = {
					M: matrix.astype( np.float32 ),
					label_SAT: ( satisfiability.astype( np.float32 ) - 0.5 ) * 2
				}
			)
		):
			print( "{}: {}".format( name, thing ) )
		#end for
	#end session
pass
