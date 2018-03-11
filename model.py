import tensorflow as tf

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

def swap(x):
	"""Swaps the lines representing the literals with the ones that represent their negated versions in a matrix.
	
  Raises:
		ValueError: if the number of dimensions is not 2 or 3."""
	two = tf.constant( 2 )
	s = x.shape
	if len(s) == 2:
		N, _ = s
		x0 = x[0:N]
		x1 = x[N:two*N]
		return tf.concat([x1,x0],0)
	elif len(s) == 3:
		_, N, _ = s
		N = int( N )
		x0 = x[:,0:N]
		x1 = x[:,N:two*N]
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
