import tensorflow as tf
import numpy as np
from collections.abc import Sequence

class Mlp(object):
	def __init__(
		self,
		layer_sizes,
		output_size = None,
		activations = None,
		output_activation = None,
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
		# If object isn't a list, assume it is a single value that will be repeated for all values
		if not isinstance( activations, list ):
			activations = [ activations for _ in layer_sizes ]
		#end if
		# If there is one specifically for the output, add it to the list of layers to be built
		if output_size is not None:
			layer_sizes = layer_sizes + [output_size]
			activations = activations + [output_activation]
		#end if
		for i, params in enumerate( zip( layer_sizes, activations ) ):
			size, activation = params
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

class SAT_solver(object):
	
	def __init__(self, embedding_size):
		# Hyperparameters
		self.learning_rate = 2e-5
		self.parameter_l2norm_scaling = 1e-10
		self.global_norm_gradient_clipping_ratio = 0.65
		self.embedding_size = embedding_size
		self.L_cell_activation = tf.nn.relu
		self.C_cell_activation = tf.nn.relu
		self.L_msg_activation = tf.nn.relu
		self.C_msg_activation = tf.nn.relu
		self.L_vote_activation = tf.nn.relu
		# Build the network
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
			activations = [ self.L_msg_activation for _ in range(2) ] + [ None ],
			name = "L_msg",
			name_internal_layers = True
		)
		# MLP that will decode a clause embedding as a message to the literal LSTM
		self.C_msg_MLP = Mlp(
			layer_sizes = [ self.embedding_size for _ in range(3) ],
			activations = [ self.C_msg_activation for _ in range(2) ] + [ None ],
			name = "C_msg",
			name_internal_layers = True
		)
		# MLP that will decode a literal embedding as a vote for satisfiability
		self.L_vote_MLP = Mlp(
			layer_sizes = [ self.embedding_size for _ in range(2) ],
			activations = [ self.L_vote_activation for _ in range(2) ],
			output_size = 1,
			name = "L_vote",
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
		assert_length_equal = tf.Assert( tf.equal( self.p, tf.shape( self.num_vars_on_instance )[0] ), [ self.instance_SAT, self.num_vars_on_instance ] )
		with tf.control_dependencies( [assert_length_equal] ):
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
			
			predicted_SAT = tf.TensorArray( size = self.p, dtype = tf.float32 )
			_, _, _, _, _, predicted_SAT, _ = tf.while_loop(
				self._vote_while_cond,
				self._vote_while_body,
				[ tf.constant( 0, dtype = tf.int32 ), self.p, tf.constant( 0, dtype = tf.int32 ), self.n, self.num_vars_on_instance, predicted_SAT, self.L_vote ]
			)
			self.predicted_SAT = predicted_SAT.stack()
			
			self.predict_costs = tf.nn.sigmoid_cross_entropy_with_logits( labels = self.instance_SAT, logits = self.predicted_SAT )
			self.predict_cost = tf.reduce_mean( self.predict_costs )
			self.vars_cost = tf.zeros([])
			self.tvars = tf.trainable_variables()
			for var in self.tvars:
				self.vars_cost = tf.add( self.vars_cost, tf.nn.l2_loss( var ) )
				#self.vars_cost += tf.nn.l2_loss( var )
			#end for
			self.vars_cost = tf.Print( self.vars_cost, [tf.shape( self.vars_cost )], "Vars ")
			self.predict_cost = tf.Print( self.predict_cost, [tf.shape( self.predict_cost )], "Pred ")
			self.loss = tf.add( self.predict_cost, tf.multiply( self.vars_cost, self.parameter_l2norm_scaling ) )
			#self.loss = tf.identity( self.predict_cost + self.vars_cost * self.parameter_l2norm_scaling )
			self.optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = self.learning_rate )
			self.grads, _ = tf.clip_by_global_norm( tf.gradients( self.loss, self.tvars ), self.global_norm_gradient_clipping_ratio )
			self.train_step = self.optimizer.apply_gradients( zip( self.grads, self.tvars ) )
			
			self.accuracy = tf.reduce_mean(
				tf.cast(
					tf.equal(
						tf.cast( self.instance_SAT, tf.bool ),
						tf.cast( tf.round( tf.nn.sigmoid( self.predicted_SAT ) ), tf.bool )
					)
					, tf.float32
				)
			)
		#end assert length equal
	#end _solve

	def f(self):
		return
	#end
	
	def _message_while_body(self, t, t_max, L_state, C_state):
		# Get the literal messages
		L_msg = self.L_msg_MLP( L_state.h )
		# Multiply the masks to the literal messages
		Mt_x_L_msg = tf.sparse_tensor_dense_matmul( self.Mt, L_msg )
		L_pos = tf.gather( L_state.h, tf.range( tf.constant( 0 ), self.n ) )
		# Send messages from negated literals to positive ones, and vice-versa
		L_neg = tf.gather( L_state.h, tf.range( self.n, self.l ) )
		L_inverted = tf.concat( [ L_neg, L_pos ], axis = 0 )
		# Update clauses LSTM state
		with tf.variable_scope( "C_cell" ):
			_, C_state = self.C_cell( inputs = Mt_x_L_msg, state = C_state )
		# end C_cell scope
		
		# Get the clause messages
		C_msg = self.C_msg_MLP( C_state.h )
		# Multiply the masks to the clause messages
		M_x_C_msg = tf.sparse_tensor_dense_matmul( self.M, C_msg )
		# Send messages from negated literals to positive ones, and vice-versa
		L_pos = tf.gather( L_state.h, tf.range( tf.constant( 0 ), self.n ) )
		L_neg = tf.gather( L_state.h, tf.range( self.n, self.l ) )
		L_inverted = tf.concat( [ L_neg, L_pos ], axis = 0 )
		# Update literal LSTM state
		with tf.variable_scope( "L_cell" ):
			_, L_state = self.L_cell( inputs = tf.concat( [ M_x_C_msg, L_inverted ], axis = 1 ), state = L_state )
		# end L_cell scope
		
		return tf.add( t, tf.constant( 1 ) ), t_max, L_state, C_state
	#end _message_while_body
	
	def _message_while_cond(self, t, t_max, L_state, C_state):
		return tf.less( t, t_max )
	#end _message_while_cond
	
	def _vote_while_body(self, i, p, n_acc, n, n_var_list, predicted_SAT, L_vote):
		# Helper for the amound of variables in this problem
		i_n = n_var_list[i]
		# Gather the positive and negative literals for that problem
		pos_lits = tf.gather( L_vote, tf.range( n_acc, tf.add( n_acc, i_n ) ) )
		neg_lits = tf.gather( L_vote, tf.range( tf.add( n, n_acc ), tf.add( n, tf.add( n_acc, i_n ) ) ) )
		# Concatenate positive and negative literals and average their vote values
		problem_predicted_SAT = tf.reduce_mean( tf.concat( [pos_lits, neg_lits], axis = 1 ) )
		# Update TensorArray
		predicted_SAT = predicted_SAT.write( i, problem_predicted_SAT )
		return tf.add( i, tf.constant( 1 ) ), p, tf.add( n_acc, i_n ), n, n_var_list, predicted_SAT, L_vote
	#end _message_while_body
	
	def _vote_while_cond(self, i, p, n_acc, n, n_var_list, predicted_sat, L_vote):
		return tf.less( i, p )
	#end _message_while_cond

	def f(self):
		return
	#end	

#end SAT_solver
