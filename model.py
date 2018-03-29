import tensorflow as tf
import numpy as np

def mlp(
	inputs,
	layer_sizes,
	output_size = None,
	activation = None,
	use_bias = True,
	kernel_initializer = None,
	bias_initializer = tf.zeros_initializer(),
	kernel_regularizer = None,
	bias_regularizer = None,
	activity_regularizer = None,
	kernel_constraint = None,
	bias_constraint = None,
	trainable = True,
	name = None,
	name_internal_layers = True,
	reuse = None
):
	"""Stacks len(layer_sizes) dense layers on top of each other, with an additional layer with output_size neurons, if specified."""
	layers = [ inputs ]
	internal_name = None
	if output_size is not None:
		layer_sizes = layer_sizes + [output_size]
	#end if
	for i, size in enumerate( layer_sizes ):
		if name_internal_layers:
			internal_name = name + "_MLP_layer_{}".format( i + 1 )
		#end if
		new_layer = tf.layers.dense(
			layers[-1],
			size,
			activation = activation,
			use_bias = use_bias,
			kernel_initializer = kernel_initializer,
			bias_initializer = bias_initializer,
			kernel_regularizer = kernel_regularizer,
			bias_regularizer = bias_regularizer,
			activity_regularizer = activity_regularizer,
			kernel_constraint = kernel_constraint,
			bias_constraint = bias_constraint,
			trainable = trainable,
			name = internal_name,
			reuse = reuse	
		)
		layers.append( new_layer )
	#end for
	return tf.identity( layers[-1], name = name )
#end mlp

class Mlp(object):
	def __init__(
		self,
		layer_sizes,
		output_size = None,
		activation = None,
		use_bias = True,
		kernel_initializer = None,
		bias_initializer = tf.zeros_initializer(),
		kernel_regularizer = None,
		bias_regularizer = None,
		activity_regularizer = None,
		kernel_constraint = None,
		bias_constraint = None,
		trainable = True,
		name = None,
		name_internal_layers = True
	):
		"""Stacks len(layer_sizes) dense layers on top of each other, with an additional layer with output_size neurons, if specified."""
		self.layers = []
		internal_name = None
		if output_size is not None:
			layer_sizes = layer_sizes + [output_size]
		#end if
		for i, size in enumerate( layer_sizes ):
			if name_internal_layers:
				internal_name = name + "_MLP_layer_{}".format( i + 1 )
			#end if
			new_layer = tf.layers.Dense(
				size,
				activation = activation,
				use_bias = use_bias,
				kernel_initializer = kernel_initializer,
				bias_initializer = bias_initializer,
				kernel_regularizer = kernel_regularizer,
				bias_regularizer = bias_regularizer,
				activity_regularizer = activity_regularizer,
				kernel_constraint = kernel_constraint,
				bias_constraint = bias_constraint,
				trainable = trainable,
				name = internal_name
			)
			self.layers.append( new_layer )
		#end for
	#end __init__
	
	def __call__( self, inputs, *args, **kwargs ):
		outputs = [ inputs ]
		for layer in self.layers:
			outputs.append( layer( outputs[-1] ) )
		#end for
		return outputs[-1]
	#end __call__
#end Mlp

def swap(x):
	"""Swaps the lines representing the literals with the ones that represent their negated versions in a matrix.
	
  Raises:
		ValueError: if the number of dimensions is not 2 or 3."""
	two = tf.constant( 2 )
	s = x.shape
	if len(s) == 2:
		N, _ = s
		x0 = x[0:two*N:2]
		x1 = x[1:two*N:2]
		return tf.concat([x1,x0],0)
	elif len(s) == 3:
		_, N, _ = s
		N = int( N )
		x0 = x[:,0:two*N:2]
		x1 = x[:,1:two*N:2]
		return tf.concat([x1,x0],1)
	else:
		raise ValueError( "Number of dimensions not supported, must be 2 or 3 and is {}".format(len(s)) )
#end swap

def build_model(
	time_steps,
	batch_size,
	d,
	n,
	m,
	Lmsg_sizes,
	Lvote_sizes,
	Cmsg_sizes,
	vote_only_on_end = False
):
	"""Builds a model for solving SAT problems with n variables and m clauses, using embeddings of size d.
		This model will use a fixed sized batches with batch_size inputs and will run the recursive part of the code for a fixed number of steps time_steps.
		The model is based on what is described in the paper ``Learning a SAT Solver from Single-Bit Supervision'', from Selsam et al., available in: https://arxiv.org/abs/1802.03685
		
	Args:
		time_steps: The number of time steps the model will be ran.
		batch_size: The number of SAT instances that will be present in each batch.
		d: The size of the embedding to be used.
		n: The number of variables in the SAT instances.
		m: The number of clauses in the SAT instances.
		Lmsg_sizes: A list containing the number of neurons for each of the layers in the Lmsg MLP.
		Lvote_sizes: A list containing the number of neurons for each of the layers in the Lvote MLP.
		Cmsg_sizes: A list containing the number of neurons for each of the layers in the Cmsg MLP
		vote_only_on_end: Whether to build the graph to vote on every timestep or only on the final one
	 Returns:
		A 6-uple (M,predicted_SAT,instance_SAT,loss,train_step,var_dict), where:
			M: The tensorflow input placeholder of shape (batch_size, 2*n, m) containing the input matrices for the model that specifies the SAT instance for each batch.
			predicted_SAT: The tensorflow handle to run the model and return the predicted value for the satisfiability with a shape (batch_size,) and values between -1 and 1, being that 1 is a high confidence that the problem is SAT and -1 a high confidence that the problem is UNSAT.
			instance_SAT: The tensorflow placeholder of shape (batch_size,) and values between -1 and 1, being that 1 means the problem i is SAT and -1 a high confidence that the problem is UNSAT.
			loss: The loss function calculated given the predicted predicted_SAT and the real instance_SAT values.
			train_step: The tensorflow handle to apply the optimizer given the input matrices and the satisfiability of the instances.
			var_dict: The dictionary that may contain additional handles to internal values of the network.
	"""
	# Sizes for the MLPs
	# Input matrix for each of the batch's SAT problem and its transposed
	M = tf.placeholder( tf.float32, [ batch_size, 2*n, m ], name = "M" )
	Mt = tf.transpose( M, [0,2,1], name = "Mt" )
	# Whether that batch's SAT problem is SAT or UNSAT
	instance_SAT = tf.placeholder( tf.float32, [ batch_size, ], name = "instance_SAT" )
	# Embedding variables
	L0 = tf.Variable( tf.random_uniform( [d], dtype = tf.float32 ), dtype = tf.float32, name = "L0" )
	C0 = tf.Variable( tf.random_uniform( [d], dtype = tf.float32 ), dtype = tf.float32, name = "C0" )
	# LSTM cells
	Lu_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
		2*n*d,
		reuse = tf.AUTO_REUSE
	)
	Cu_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
		m*d,
		reuse = tf.AUTO_REUSE
	)
	# Starting states for the LSTM cells
	Lu_cell_init_hidden_state = Lu_cell.zero_state( batch_size, dtype = tf.float32 )
	Cu_cell_init_hidden_state = Cu_cell.zero_state( batch_size, dtype = tf.float32 )
 

	# Building the unrolled graph
	current_L = tf.reshape(
		tf.tile(
			L0,
			(batch_size*2*n,),
			name = "L0_tiled_to_fit"
		),
		(batch_size,2*n,d),
		name = "L"
	)
	current_Lh = Lu_cell_init_hidden_state
	current_C = tf.reshape(
		tf.tile(
			C0,
			(batch_size*m,),
			name = "C0_tiled_to_fit"
		),
		(batch_size,m,d),
		name = "C"
	)
	current_Ch = Cu_cell_init_hidden_state
	L = []
	Lh = []
	Lm = []
	Lv = []
	C = []
	Ch = []
	Cm = []
	# For each time step
	for t in range( time_steps ):
		# Get the values for Lmsg, Cmsg and Lvote
		L_flat = tf.reshape(
			current_L,
			[ batch_size, -1 ],
			name = "L_flat"
		)
		Lmsg_flat = mlp(
			L_flat,
			Lmsg_sizes,
			output_size = 2 * n * d,
			activation = tf.nn.relu,
			name = "Lmsg",
			reuse = tf.AUTO_REUSE # Reuse the same layer weigths for other time_steps
		)
		Lmsg = tf.reshape(
			 Lmsg_flat,
			(batch_size, 2*n, d),
			name = "Lmsg_reshaped"
		)
		C_flat = tf.reshape(
			current_C,
			[ batch_size, -1 ],
			name = "C_flat"
		)
		Cmsg_flat = mlp(
			C_flat,
			Cmsg_sizes,
			output_size = m * d,
			activation = tf.nn.relu,
			name = "Cmsg",
			reuse = tf.AUTO_REUSE # Reuse the same layer weigths for other time_steps
		)
		Cmsg = tf.reshape(
			Cmsg_flat,
			(batch_size, m, d),
			name = "Cmsg_reshaped"
		)
		if not vote_only_on_end or t + 1 >= time_steps:
			Lvote = mlp(
				L_flat,
				Lvote_sizes,
				output_size = 2 * n,
				activation = tf.nn.tanh,
				name = "Lvote",
				reuse = tf.AUTO_REUSE # Reuse the same layer weigths for other time_steps
			)
		else:
			Lvote = None
		#end if
		# Get the input values for Lu and Cu

		Cin = tf.matmul( Mt, Lmsg, name = "Cin" )
		Cin_flat = tf.reshape( Cin, (batch_size, m*d), name = "Cin_flat" )
		Lin = tf.concat(
			[
				current_L,
				tf.matmul(
					M,
					Cmsg,
					name = "M_x_Cmsg"
				)
			],
			axis = 1,
			name = "Lin"
		)
		Lin_flat = tf.reshape(
			Lin,
			(batch_size, 2*(2*n)*d),
			name = "Lin_flat"
		)

		# Run the inputs and last states through the cells
		with tf.variable_scope( "Cu_cell", reuse = tf.AUTO_REUSE ): # Theoretically already being reused
			new_C_flat, new_Ch = Cu_cell(
				Cin_flat,
				current_Ch
			)
		with tf.variable_scope( "Lu_cell", reuse = tf.AUTO_REUSE ):
			new_L_flat, new_Lh = Lu_cell(
				Lin_flat,
				current_Lh
			)
		new_L = tf.reshape(
			new_L_flat,
			[batch_size,2*n,d],
			name = "L"
		)
		new_C = tf.reshape(
			new_C_flat,
			[batch_size,m,d],
			name = "C"
		)
		# Append the values into a list, for bookkeeping
		L.append( new_L )
		Lh.append( new_Lh )
		Lm.append( Lmsg )
		Lv.append( Lvote )
		C.append( new_C )
		Ch.append( new_Ch )
		Cm.append( Cmsg )
		# Update current values
		current_L = new_L
		current_Lh = new_Lh
		current_C = new_C
		current_Ch = new_Ch
	#end for
	# Predict whether the instance is SAT for every instance in the batch
	predicted_SAT = tf.reduce_mean(
		Lv[-1],
		axis = 1,
		name = "predicted_SAT"
	)
	loss = tf.losses.mean_squared_error( instance_SAT, predicted_SAT )	
	train_step = tf.train.AdamOptimizer( name = "Adam" ).minimize( loss )
	var_dict = {
		"L": L,
		"Lh": Lh,
		"L_msg": Lm,
		"L_vote": Lv,
		"C": C,
		"Ch": Ch,
		"C_msg": Cm,
		"Trainable vars": tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	}
	return M, predicted_SAT, instance_SAT, loss, train_step, var_dict
#end build_model

def build_model_while(
	batch_size,
	d,
	n,
	m,
	Lmsg_sizes,
	Lvote_sizes,
	Cmsg_sizes,
	vote_only_on_end = False
):
	"""Builds a model for solving SAT problems with n variables and m clauses, using embeddings of size d.
		This model will use a fixed sized batches with batch_size inputs and will run the recursive part of the code based on a placeholder, returned by this function.
		The model is based on what is described in the paper ``Learning a SAT Solver from Single-Bit Supervision'', from Selsam et al., available in: https://arxiv.org/abs/1802.03685
		
	Args:
		batch_size: The number of SAT instances that will be present in each batch.
		d: The size of the embedding to be used.
		n: The number of variables in the SAT instances.
		m: The number of clauses in the SAT instances.
		Lmsg_sizes: A list containing the number of neurons for each of the layers in the Lmsg MLP.
		Lvote_sizes: A list containing the number of neurons for each of the layers in the Lvote MLP.
		Cmsg_sizes: A list containing the number of neurons for each of the layers in the Cmsg MLP
		vote_only_on_end: Whether to build the graph to vote on every timestep or only on the final one
	 Returns:
		A 7-uple (M,predicted_SAT,instance_SAT,loss,train_step,var_dict), where:
			M: The tensorflow input placeholder of shape (batch_size, 2*n, m) containing the input matrices for the model that specifies the SAT instance for each batch.
			time_steps: The tensorflow input placeholder of shape () that indicates how many steps should the network unroll
			predicted_SAT: The tensorflow handle to run the model and return the predicted value for the satisfiability with a shape (batch_size,) and values between -1 and 1, being that 1 is a high confidence that the problem is SAT and -1 a high confidence that the problem is UNSAT.
			instance_SAT: The tensorflow placeholder of shape (batch_size,) and values between -1 and 1, being that 1 means the problem i is SAT and -1 a high confidence that the problem is UNSAT.
			loss: The loss function calculated given the predicted predicted_SAT and the real instance_SAT values.
			train_step: The tensorflow handle to apply the optimizer given the input matrices and the satisfiability of the instances.
			var_dict: The dictionary that may contain additional handles to internal values of the network.
	"""
	# Sizes for the MLPs
	# Input matrix for each of the batch's SAT problem and its transposed
	time_steps = tf.placeholder( tf.int32, shape = (), name = "time_steps" )
	M = tf.placeholder( tf.float32, [ batch_size, 2*n, m ], name = "M" )
	Mt = tf.transpose( M, [0,2,1], name = "Mt" )
	# Whether that batch's SAT problem is SAT or UNSAT
	instance_SAT = tf.placeholder( tf.float32, [ batch_size, ], name = "instance_SAT" )
	# Embedding variables
	L0 = tf.Variable( tf.random_uniform( [d], dtype = tf.float32 ), dtype = tf.float32, name = "L0" )
	C0 = tf.Variable( tf.random_uniform( [d], dtype = tf.float32 ), dtype = tf.float32, name = "C0" )
	# LSTM cells
	Lu_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
		2*n*d,
		reuse = tf.AUTO_REUSE
	)
	Cu_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
		m*d,
		reuse = tf.AUTO_REUSE
	)
	# Starting states for the LSTM cells
	Lu_cell_init_hidden_state = Lu_cell.zero_state( batch_size, dtype = tf.float32 )
	Cu_cell_init_hidden_state = Cu_cell.zero_state( batch_size, dtype = tf.float32 )

	# Initializing variables for the loop
	current_L = tf.reshape(
		tf.tile(
			L0,
			(batch_size*2*n,),
			name = "L0_tiled_to_fit"
		),
		(batch_size,2*n,d),
		name = "L"
	)
	current_Lh = Lu_cell_init_hidden_state
	current_C = tf.reshape(
		tf.tile(
			C0,
			(batch_size*m,),
			name = "C0_tiled_to_fit"
		),
		(batch_size,m,d),
		name = "C"
	)
	current_Ch = Cu_cell_init_hidden_state

	Lvote = tf.ones(
		[batch_size, 2 * n],
		dtype = tf.float32,
		name = "Dummy_Lvote"
	)
	
	# Define the condition
	def condition( i, time_steps, current_L, current_Lh, current_C, current_Ch, Lvote ):
		return tf.less( i, time_steps )
	#end condition
	
	# Define the loop body
	def loop_body( i, time_steps, current_L, current_Lh, current_C, current_Ch, Lvote ):
		# Get the values for Lmsg, Cmsg and Lvote
		L_flat = tf.reshape(
			current_L,
			[ batch_size, -1 ],
			name = "L_flat"
		)
		Lmsg_flat = mlp(
			L_flat,
			Lmsg_sizes,
			output_size = 2 * n * d,
			activation = tf.nn.relu,
			name = "Lmsg",
			reuse = tf.AUTO_REUSE # Reuse the same layer weigths for other time_steps
		)
		Lmsg = tf.reshape(
			 Lmsg_flat,
			(batch_size, 2*n, d),
			name = "Lmsg_reshaped"
		)
		C_flat = tf.reshape(
			current_C,
			[ batch_size, -1 ],
			name = "C_flat"
		)
		Cmsg_flat = mlp(
			C_flat,
			Cmsg_sizes,
			output_size = m * d,
			activation = tf.nn.relu,
			name = "Cmsg",
			reuse = tf.AUTO_REUSE
		)
		Cmsg = tf.reshape(
			Cmsg_flat,
			(batch_size, m, d),
			name = "Cmsg_reshaped"
		)
		Lvote = mlp(
			L_flat,
			Lvote_sizes,
			output_size = 2 * n,
			activation = tf.nn.tanh,
			name = "Lvote",
			reuse = tf.AUTO_REUSE
		)

		# Get the input values for Lu and Cu
		Cin = tf.matmul( Mt, Lmsg, name = "Cin" )
		Cin_flat = tf.reshape( Cin, (batch_size, m*d), name = "Cin_flat" )
		Lin = tf.concat(
			[
				current_L,
				tf.matmul(
					M,
					Cmsg,
					name = "M_x_Cmsg"
				)
			],
			axis = 1,
			name = "Lin"
		)
		Lin_flat = tf.reshape(
			Lin,
			(batch_size, 2*(2*n)*d),
			name = "Lin_flat"
		)

		# Run the inputs and last states through the cells
		with tf.variable_scope( "Cu_cell", reuse = tf.AUTO_REUSE ):
			new_C_flat, new_Ch = Cu_cell(
				Cin_flat,
				current_Ch
			)
		with tf.variable_scope( "Lu_cell", reuse = tf.AUTO_REUSE ):
			new_L_flat, new_Lh = Lu_cell(
				Lin_flat,
				current_Lh
			)
		new_L = tf.reshape(
			new_L_flat,
			[batch_size,2*n,d],
			name = "L"
		)
		new_C = tf.reshape(
			new_C_flat,
			[batch_size,m,d],
			name = "C"
		)
		
		# Update current values
		return tf.add( i, tf.constant(1), name = "Increment" ), time_steps, new_L, new_Lh, new_C, new_Ch, Lvote 
	#end condition

	# Build/run the loop in tensorflow
	last_i, _, last_L, last_Lh, last_C, last_Ch, last_Lvote = tf.while_loop(
		condition,
		loop_body,
		(tf.constant( 0, name = "loop_counter_init_value" ), time_steps, current_L, current_Lh, current_C, current_Ch, Lvote)
	)

	# Predict whether the instance is SAT for every instance in the batch
	predicted_SAT = tf.reduce_mean(
		last_Lvote,
		axis = 1,
		name = "predicted_SAT"
	)
	loss = tf.losses.mean_squared_error( instance_SAT, predicted_SAT )	
	train_step = tf.train.AdamOptimizer( name = "Adam" ).minimize( loss )
	var_dict = {
		"L": last_L,
		"Lh": last_Lh,
		"L_vote": last_Lvote,
		"C": last_C,
		"Ch": last_Ch,
		"Trainable vars": tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	}
	return M, time_steps, predicted_SAT, instance_SAT, loss, train_step, var_dict
#end build_model_while


def build_model_sparse_while_no_batch(
	d,
	n,
	m,
	Lmsg_sizes,
	Lvote_sizes,
	Cmsg_sizes,
	vote_only_on_end = False
):
	"""Builds a model for solving SAT problems with n variables and m clauses, using embeddings of size d.
		This model will use a fixed sized batches with batch_size inputs and will run the recursive part of the code based on a placeholder, returned by this function.
		The model is based on what is described in the paper ``Learning a SAT Solver from Single-Bit Supervision'', from Selsam et al., available in: https://arxiv.org/abs/1802.03685
		
	Args:
		batch_size: The number of SAT instances that will be present in each batch.
		d: The size of the embedding to be used.
		n: The number of variables in the SAT instances.
		m: The number of clauses in the SAT instances.
		Lmsg_sizes: A list containing the number of neurons for each of the layers in the Lmsg MLP.
		Lvote_sizes: A list containing the number of neurons for each of the layers in the Lvote MLP.
		Cmsg_sizes: A list containing the number of neurons for each of the layers in the Cmsg MLP
		vote_only_on_end: Whether to build the graph to vote on every timestep or only on the final one
	 Returns:
		A 7-uple (M,predicted_SAT,instance_SAT,loss,train_step,var_dict), where:
			M: The tensorflow input placeholder of shape (batch_size, 2*n, m) containing the input matrices for the model that specifies the SAT instance for each batch.
			time_steps: The tensorflow input placeholder of shape () that indicates how many steps should the network unroll
			predicted_SAT: The tensorflow handle to run the model and return the predicted value for the satisfiability with a shape (batch_size,) and values between -1 and 1, being that 1 is a high confidence that the problem is SAT and -1 a high confidence that the problem is UNSAT.
			instance_SAT: The tensorflow placeholder of shape (batch_size,) and values between -1 and 1, being that 1 means the problem i is SAT and -1 a high confidence that the problem is UNSAT.
			loss: The loss function calculated given the predicted predicted_SAT and the real instance_SAT values.
			train_step: The tensorflow handle to apply the optimizer given the input matrices and the satisfiability of the instances.
			var_dict: The dictionary that may contain additional handles to internal values of the network.
	"""
	# Sizes for the MLPs
	batch_size = 1
	# Input matrix for each of the batch's SAT problem and its transposed
	time_steps = tf.placeholder( tf.int32, shape = (), name = "time_steps" )
	M = tf.sparse_placeholder( tf.float32, shape = [ None, m ], name = "M" )
	Mt = tf.sparse_transpose( M, [1,0], name = "Mt" )
	#M = tf.sparse_reshape( Ms, [ 1, 2*n, m ] )
	#Mt = tf.sparse_transpose( M, [0,2,1], name = "Mt" )
	# Whether that batch's SAT problem is SAT or UNSAT
	instance_SAT = tf.placeholder( tf.float32, [ batch_size, ], name = "instance_SAT" )
	# Embedding variables
	L0 = tf.Variable( tf.random_uniform( [d], dtype = tf.float32 ), dtype = tf.float32, name = "L0" )
	C0 = tf.Variable( tf.random_uniform( [d], dtype = tf.float32 ), dtype = tf.float32, name = "C0" )
	# LSTM cells
	Lu_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
		2*n*d,
		reuse = tf.AUTO_REUSE
	)
	Cu_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
		m*d,
		reuse = tf.AUTO_REUSE
	)
	# Starting states for the LSTM cells
	Lu_cell_init_hidden_state = Lu_cell.zero_state( batch_size, dtype = tf.float32 )
	Cu_cell_init_hidden_state = Cu_cell.zero_state( batch_size, dtype = tf.float32 )

	# Initializing variables for the loop
	current_L = tf.reshape(
		tf.tile(
			L0,
			(batch_size*2*n,),
			name = "L0_tiled_to_fit"
		),
		(batch_size,2*n,d),
		name = "L"
	)
	current_Lh = Lu_cell_init_hidden_state
	current_C = tf.reshape(
		tf.tile(
			C0,
			(batch_size*m,),
			name = "C0_tiled_to_fit"
		),
		(batch_size,m,d),
		name = "C"
	)
	current_Ch = Cu_cell_init_hidden_state

	Lvote = tf.ones(
		[batch_size, 2 * n],
		dtype = tf.float32,
		name = "Dummy_Lvote"
	)
	
	# Define the condition
	def condition( i, time_steps, current_L, current_Lh, current_C, current_Ch, Lvote ):
		return tf.less( i, time_steps )
	#end condition
	
	# Define the loop body
	def loop_body( i, time_steps, current_L, current_Lh, current_C, current_Ch, Lvote ):
		# Get the values for Lmsg, Cmsg and Lvote
		L_flat = tf.reshape(
			current_L,
			[ batch_size, -1 ],
			name = "L_flat"
		)
		Lmsg_flat = mlp(
			L_flat,
			Lmsg_sizes,
			output_size = 2 * n * d,
			activation = tf.nn.relu,
			name = "Lmsg",
			reuse = tf.AUTO_REUSE # Reuse the same layer weigths for other time_steps
		)
		Lmsg = tf.reshape(
			 Lmsg_flat,
			#(batch_size, 2*n, d),
			( 2*n, d ),
			name = "Lmsg_reshaped"
		)
		C_flat = tf.reshape(
			current_C,
			[ batch_size, -1 ],
			name = "C_flat"
		)
		Cmsg_flat = mlp(
			C_flat,
			Cmsg_sizes,
			output_size = m * d,
			activation = tf.nn.relu,
			name = "Cmsg",
			reuse = tf.AUTO_REUSE
		)
		Cmsg = tf.reshape(
			Cmsg_flat,
			#(batch_size, m, d),
			( m, d ),
			name = "Cmsg_reshaped"
		)
		Lvote = mlp(
			L_flat,
			Lvote_sizes,
			output_size = 2 * n,
			activation = tf.nn.tanh,
			name = "Lvote",
			reuse = tf.AUTO_REUSE
		)

		# Get the input values for Lu and Cu
		Cin = tf.expand_dims( tf.sparse_tensor_dense_matmul( Mt, Lmsg, name = "Cin" ), 0 )
		Cin_flat = tf.reshape( Cin, (batch_size, m*d), name = "Cin_flat" )
		Lin = tf.concat(
			[
				current_L,
				tf.expand_dims(
					tf.sparse_tensor_dense_matmul(
						M,
						Cmsg,
						name = "M_x_Cmsg"
					),
					0
				)
			],
			axis = 1,
			name = "Lin"
		)
		Lin_flat = tf.reshape(
			Lin,
			(batch_size, 2*(2*n)*d),
			name = "Lin_flat"
		)

		# Run the inputs and last states through the cells
		with tf.variable_scope( "Cu_cell", reuse = tf.AUTO_REUSE ):
			new_C_flat, new_Ch = Cu_cell(
				Cin_flat,
				current_Ch
			)
		with tf.variable_scope( "Lu_cell", reuse = tf.AUTO_REUSE ):
			new_L_flat, new_Lh = Lu_cell(
				Lin_flat,
				current_Lh
			)
		new_L = tf.reshape(
			new_L_flat,
			[batch_size,2*n,d],
			name = "L"
		)
		new_C = tf.reshape(
			new_C_flat,
			[batch_size,m,d],
			name = "C"
		)
		
		# Update current values
		return tf.add( i, tf.constant(1), name = "Increment" ), time_steps, new_L, new_Lh, new_C, new_Ch, Lvote 
	#end condition

	# Build/run the loop in tensorflow
	last_i, _, last_L, last_Lh, last_C, last_Ch, last_Lvote = tf.while_loop(
		condition,
		loop_body,
		(tf.constant( 0, name = "loop_counter_init_value" ), time_steps, current_L, current_Lh, current_C, current_Ch, Lvote)
	)

	# Predict whether the instance is SAT for every instance in the batch
	predicted_SAT = tf.reduce_mean(
		last_Lvote,
		axis = 1,
		name = "predicted_SAT"
	)
	loss = tf.losses.mean_squared_error( instance_SAT, predicted_SAT )	
	train_step = tf.train.AdamOptimizer( name = "Adam" ).minimize( loss )
	var_dict = {
		"L": last_L,
		"Lh": last_Lh,
		"L_vote": last_Lvote,
		"C": last_C,
		"Ch": last_Ch,
		"Trainable vars": tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	}
	return M, time_steps, predicted_SAT, instance_SAT, loss, train_step, var_dict
#end build_model_while

class SAT_solver(object):
	
	def __init__(self, embedding_size):
		self.embedding_size = embedding_size
		self.L_cell_activation = tf.nn.tanh
		self.C_cell_activation = tf.nn.tanh
		self.L_msg_activation = tf.nn.relu
		self.C_msg_activation = tf.nn.relu
		self.L_vote_activation = tf.nn.tanh
		with tf.variable_scope( "SAT_solver" ):
			with tf.variable_scope( "placeholders" ) as scope:
				self._init_placeholders()
			#end placeholder scope
			with tf.variable_scope( "parameters" ) as scope:
				self._init_parameters()
			with tf.variable_scope( "utilities" ) as scope:
				self._init_util_vars()
			with tf.variable_scope( "solve" ) as scope:
				self._solve()
			#end solve scope
		#end SAT_solver scope
	#end __init__

	def _init_placeholders(self): 
		self.time_steps = tf.placeholder( tf.int32, shape = (), name = "time_steps" )
		self.M = tf.sparse_placeholder( tf.float32, shape = [ None, None ], name = "M" )
		self.instance_SAT = tf.placeholder( tf.float32, [ None ], name = "instance_SAT" )
		self.num_vars_on_instance = tf.placeholder( tf.int32, [ None ], name = "instance_n" )
	#end _init_placeholders
	
	def _init_parameters(self):
		# Iniitial Literal Embedding
		self.L_init = tf.get_variable(
			"L_init",
			[ 1, self.embedding_size ],
			dtype = tf.float32
		)
		# Iniitial Clause Embedding
		self.C_init = tf.get_variable(
			"C_init",
			[ 1, self.embedding_size ],
			dtype = tf.float32
		)
		# LSTM Cell that will produce literal embeddings
		self.L_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
			self.embedding_size,
			activation = self.L_cell_activation
		)
		# LSTM Cell that will produce clause embeddings
		self.C_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
			self.embedding_size,
			activation = self.C_cell_activation
		)
		# MLP that will decode a literal embedding as a message to the clause LSTM
		self.L_msg_MLP = Mlp(
			layer_sizes = [ self.embedding_size for _ in range(3) ],
			name = "L_msg",
			activation = self.L_msg_activation,
			name_internal_layers = True
		)
		# MLP that will decode a clause embedding as a message to the literal LSTM
		self.C_msg_MLP = Mlp(
			layer_sizes = [ self.embedding_size for _ in range(3) ],
			name = "C_msg",
			activation = self.C_msg_activation,
			name_internal_layers = True
		)
		# MLP that will decode a literal embedding as a vote for satisfiability
		self.L_vote_MLP = Mlp(
			layer_sizes = [ self.embedding_size for _ in range(2) ],
			output_size = 1,
			name = "L_vote",
			activation = self.L_vote_activation,
			name_internal_layers = True
		)
		return
	#end _init_parameters
	
	def _init_util_vars(self):
		self.Mt = tf.sparse_transpose( self.M, [1,0], name = "Mt" )
		self.l = tf.shape( self.M )[0]
		self.n = tf.floordiv( self.l, tf.constant( 2 ) )
		self.m = tf.shape( self.M )[1]
		self.p = tf.shape( self.instance_SAT )[0]
	#end _init_util_vars
	
	def _solve(self):
		# TODO dependency scope to eliminate errors
		pass
		# Prepare the LSTM tuple for the starting state of the literal LSTM
		L_cell_h0 = tf.tile( self.L_init , [ self.l, 1 ] )
		L_cell_c0 = tf.zeros_like( L_cell_h0, dtype = tf.float32 )
		L_state = tf.contrib.rnn.LSTMStateTuple( h = L_cell_h0, c = L_cell_c0 )
		# Prepare the LSTM tuple for the starting state of the clause LSTM
		C_cell_h0 = tf.tile( self.C_init , [ self.m, 1 ] )
		C_cell_c0 = tf.zeros_like( C_cell_h0, dtype = tf.float32 )
		C_state = tf.contrib.rnn.LSTMStateTuple( h = C_cell_h0, c = C_cell_c0 )
		# Run self.time_steps iterations of message-passing
		_, _, L_state, C_state = tf.while_loop(
			self._message_while_cond,
			self._message_while_body,
			[ tf.constant(0), self.time_steps, L_state, C_state ]
		)
		# Get the last embeddings
		self.L_n = L_state.h
		self.C_n = C_state.h
		self.L_vote = self.L_vote_MLP( self.L_n )
		
		predicted_sat = tf.TensorArray( size = self.p, dtype = tf.float32 )
		_, _, _, _, _, predicted_sat, _ = tf.while_loop(
			self._vote_while_cond,
			self._vote_while_body,
			[ tf.constant( 0, dtype = tf.int32 ), self.p, tf.constant( 0, dtype = tf.int32 ), self.n, self.num_vars_on_instance, predicted_sat, self.L_vote ]
		)
		self.predicted_sat = predicted_sat.stack()
	#end _solve

	def f(self):
		return
	#end
	
	def _message_while_body(self, t, t_max, L_state, C_state):
		# Get the messages
		L_msg = self.L_msg_MLP( L_state.h )
		C_msg = self.C_msg_MLP( C_state.h )
		# Multiply the masks to the messages
		Mt_x_L_msg = tf.sparse_tensor_dense_matmul( self.Mt, L_msg )
		M_x_C_msg = tf.sparse_tensor_dense_matmul( self.M, C_msg )
		L_pos = tf.gather( L_state.h, tf.range( tf.constant( 0 ), self.n ) )
		# Send messages from negated literals to positive ones, and vice-versa
		L_neg = tf.gather( L_state.h, tf.range( self.n, self.l ) )
		L_inverted = tf.concat( [ L_neg, L_pos ], axis = 0 )
		# Update LSTMs state
		with tf.variable_scope( "L_cell" ):
			_, L_state = self.L_cell( inputs = tf.concat( [ M_x_C_msg, L_inverted ], axis = 1 ), state = L_state )
		# end L_cell scope
		with tf.variable_scope( "C_cell" ):
			_, C_state = self.C_cell( inputs = Mt_x_L_msg, state = C_state )
		# end C_cell scope
		
		return tf.add( t, tf.constant( 1 ) ), t_max, L_state, C_state
	#end _message_while_body
	
	def _message_while_cond(self, t, t_max, L_state, C_state):
		return tf.less( t, t_max )
	#end _message_while_cond
	
	def _vote_while_body(self, i, p, n_acc, n, n_var_list, predicted_sat, L_vote):
		i_n = n_var_list[i]
		i = tf.Print( i, [i, p, tf.less( i, p )], "i")
		n_acc = tf.Print( n_acc, [n_acc, n, i_n], "n")
		pos_lits = tf.gather( L_vote, tf.range( n_acc, tf.add( n_acc, i_n ) ) )
		neg_lits = tf.gather( L_vote, tf.range( tf.add( n, n_acc ), tf.add( n, tf.add( n_acc, i_n ) ) ) )
		predicted_sat = predicted_sat.write( i, tf.reduce_mean( tf.concat( [pos_lits, neg_lits], axis = 1 ) ) )
		return tf.add( i, tf.constant( 1 ) ), p, tf.add( n_acc, i_n ), n, n_var_list, predicted_sat, L_vote
	#end _message_while_body
	
	def _vote_while_cond(self, i, p, n_acc, n, n_var_list, predicted_sat, L_vote):
		return tf.less( i, p )
	#end _message_while_cond

	def f(self):
		return
	#end	

#end SAT_solver


if __name__ == "__main__":
	solver = SAT_solver( 10 )
	feed_dict = {
		solver.time_steps: 10,
		solver.M: (
			np.array(
				[
					[0,0],
					[1,1]
				],
				dtype = np.float32
			),
			np.array(
				[
					1,
					1
				],
				dtype = np.float32
			),
			np.array( [9,20], dtype = np.int32 )
		),
		solver.instance_SAT: [1,-1,1],
		solver.num_vars_on_instance: [2,3,4] 
	}
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )
		print(
			sess.run(
				[ solver.predicted_sat ],
				feed_dict = feed_dict
			)
		)
	#end session
#end main
