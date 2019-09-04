# So.  this is a work-in-progress.  the basic idea is that this is a neural network that uses 
# stateful LSTM to process streaming video from a webcam, and attempts to predict what it will see next
# at some small timestep into the future.



#TODO: use tensorboard to watch things. --DONE! ...-ish
#TODO: take coords out of vid_in and stick in self_output --DONE!
#TODO: lag self_output --IT WAS ALREADY DONE!
#TODO: add color toggle --DONE!
#TODO: add lag in target for pred --DONE!
#TOOD: add in color functionality --DONE!
#TODO: fix the session timeout bug  --DONE!

#TODO: clean stuff up. !
#TODO: add convolutional layers ~!
#TODO: full stateless lstm
#TODO: improve speed of weight update aggregation
#TODO: connect to spark core
#TODO: compare preds to v_in
#TODO: implement save/load functionality
#TODO: break apart updown_leftright into four parameters instead of two
#TODO: add confidence gating to prediction outputs


# https://jhui.github.io/2017/03/12/TensorBoard-visualize-your-learning/ is a nice little reference to using 
# tensorboard

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# Training Parameters

logs_dir = '/home/miej/projects/4/logs/'
webcam_1_index = 1


learning_rate = 0.1  #default
adam_beta1=0.9  # default: 0.9
adam_beta2=0.999  #default: 0.999
batch_size = 1


pi=np.pi
n_step_toggle = 1000.0 # toggle for loss function every n steps.  currently not used.
display_step = 100  # prints metrics every n steps

num_video_lag_steps=8  # of frames by which to delay the objective (vs the input vision)

micsac_magnitude=2.0  # 0.0 for none 2.0 for single pixel.  (this introduces little shaking movements
                      #            especially when the model keeps its vision stationary.  goal is to help
                      #            enforce translation invariance into the learned representations)
        
with_selfmovement=1.0  # 1 or 0, whether or not we allow the neural net to control where it looks

# Network Parameters
num_hidden_1 = 128 # 1st layer num features
num_hidden_2 = 256 # 2nd layer num features 

full_video_height = 480
full_video_width = 640
full_video_channels = 3

# out of the full video dimensions, we are only feeding the network a portion of it in order to simulate
# the network having 'eye muscles' and being able to look at different things / otherwise interact with 
# what it observes.  in this case, it is able to move around its vision, but only within the confines of
# the full video frame.
video_height = 240
video_width = 320
video_channels = 1  # only 1 for grayscale or 3 for rgb please.

current_x_coord = video_width-1
current_y_coord = video_height-1

coords_dim = 2  # number of x,y coords
video_input_size = video_height * video_width * video_channels

self_output_to_user_size = 13  # for now this is something of a placeholder for potential future 
                                # capabilities for the network to influence its own surroundings.
                                # right now it does so by being able to display directionsl 'requests' 
                                #  to the user.
            
self_output_to_self_size = 2 + coords_dim  # the 2 represents the variables that allow the net to say
                                            # 'i want to move my vision up/down' or 'left/right'
    
self_output_size = self_output_to_user_size + self_output_to_self_size


def placeholder_inputs():
    # placeholders.
    
    vision_lag = tf.placeholder(tf.float32, [None, video_input_size])
    vision_target = tf.placeholder(tf.float32, [None, video_input_size])
    vision_0 = tf.placeholder(tf.float32, [None, video_input_size])
    vision_minus_1 = tf.placeholder(tf.float32, [None, video_input_size])
    vision_minus_2 = tf.placeholder(tf.float32, [None, video_input_size])
    vision_minus_3 = tf.placeholder(tf.float32, [None, video_input_size])
    vision_minus_4 = tf.placeholder(tf.float32, [None, video_input_size])
    visual_data = tf.placeholder(tf.float32, [full_video_height, full_video_width, video_channels])
    last_prediction = tf.placeholder(tf.float32, [None, video_input_size])
    
    self_output = tf.placeholder(tf.float32, [None, self_output_size])
    
    input_x_coord = tf.placeholder_with_default([current_x_coord], [1])
    input_y_coord = tf.placeholder_with_default([current_y_coord], [1])
    updown_leftright = tf.placeholder_with_default([[0.0, 0.0]], [1,2])
    step_number = tf.placeholder_with_default([0.0], [1])
    last_loss = tf.placeholder(tf.float32, [None, video_input_size])
    
    return last_loss, last_prediction, step_number, vision_lag, vision_target, vision_0, vision_minus_1, vision_minus_2, vision_minus_3, vision_minus_4, self_output, input_x_coord, input_y_coord, updown_leftright, visual_data

def get_weights_biases():
    # weights, biases, and lstm state variables for the model.
    weights = {

        'in_vid_0_h1' : tf.get_variable('in_vid_0_h1', shape=(video_input_size, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_0_h2' : tf.get_variable('in_vid_0_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_1_h1' : tf.get_variable('in_vid_1_h1', shape=(video_input_size, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_1_h2' : tf.get_variable('in_vin_vid_1_h2id_0', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_2_h1' : tf.get_variable('in_vid_2_h1', shape=(video_input_size, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_2_h2' : tf.get_variable('in_vid_2_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_3_h1' : tf.get_variable('in_vid_3_h1', shape=(video_input_size, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_3_h2' : tf.get_variable('in_vid_3_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_4_h1' : tf.get_variable('in_vid_4_h1', shape=(video_input_size, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_4_h2' : tf.get_variable('in_vid_4_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'enc_1_h1' : tf.get_variable('enc_1_h1', shape=(num_hidden_2*5, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'enc_1_h2' : tf.get_variable('enc_1_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'enc_2_h1' : tf.get_variable('enc_2_h1', shape=(self_output_size+video_input_size*2, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'enc_2_h2' : tf.get_variable('enc_2_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'dec_1_h1' : tf.get_variable('dec_1_h1', shape=(num_hidden_2*2, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'dec_1_h2' : tf.get_variable('dec_1_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'dec_1_h3' : tf.get_variable('dec_1_h3', shape=(num_hidden_2, self_output_size), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'enc_3_h1' : tf.get_variable('enc_3_h1', shape=(self_output_size - coords_dim, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'enc_3_h2' : tf.get_variable('enc_3_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'dec_2_h1' : tf.get_variable('dec_2_h1', shape=(num_hidden_2*2, num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'dec_2_h2' : tf.get_variable('dec_2_h2', shape=(num_hidden_1, num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        #'dec_2_h3' : tf.get_variable('dec_2_h3', shape=(num_hidden_2, video_input_size+video_height*video_width), 
        'dec_2_h3' : tf.get_variable('dec_2_h3', shape=(num_hidden_2, video_input_size), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
    }


    biases = {
        'in_vid_0_b1' : tf.get_variable('in_vid_0_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_0_b2' : tf.get_variable('in_vid_0_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_1_b1' : tf.get_variable('in_vid_1_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_1_b2' : tf.get_variable('in_vid_1_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_2_b1' : tf.get_variable('in_vid_2_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_2_b2' : tf.get_variable('in_vid_2_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_3_b1' : tf.get_variable('in_vid_3_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_3_b2' : tf.get_variable('in_vid_3_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'in_vid_4_b1' : tf.get_variable('in_vid_4_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'in_vid_4_b2' : tf.get_variable('in_vid_4_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'enc_1_b1' : tf.get_variable('enc_1_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'enc_1_b2' : tf.get_variable('enc_1_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'enc_2_b1' : tf.get_variable('enc_2_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'enc_2_b2' : tf.get_variable('enc_2_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'dec_1_b1' : tf.get_variable('dec_1_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'dec_1_b2' : tf.get_variable('dec_1_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'dec_1_b3' : tf.get_variable('dec_1_b3', shape=(self_output_size), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'enc_3_b1' : tf.get_variable('enc_3_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'enc_3_b2' : tf.get_variable('enc_3_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),

        'dec_2_b1' : tf.get_variable('dec_2_b1', shape=(num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        'dec_2_b2' : tf.get_variable('dec_2_b2', shape=(num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
        #'dec_2_b3' : tf.get_variable('dec_2_b3', shape=(video_input_size+video_height*video_width), 
        'dec_2_b3' : tf.get_variable('dec_2_b3', shape=(video_input_size), 
                                     initializer = tf.contrib.layers.xavier_initializer()),
    }

    states = {
        # shape should have shape [batch_size, self.state_size]. state_size for lstm is 2x # hidden units
        'in_vid_0_layer_1_states_c' : tf.get_variable('in_vid_0_layer_1_states_c', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_0_layer_1_states_h' : tf.get_variable('in_vid_0_layer_1_states_h', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'in_vid_0_layer_2_states_c' : tf.get_variable('in_vid_0_layer_2_states_c', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_0_layer_2_states_h' : tf.get_variable('in_vid_0_layer_2_states_h', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'in_vid_1_layer_1_states_c' : tf.get_variable('in_vid_1_layer_1_states_c', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_1_layer_1_states_h' : tf.get_variable('in_vid_1_layer_1_states_h', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'in_vid_1_layer_2_states_c' : tf.get_variable('in_vid_1_layer_2_states_c', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_1_layer_2_states_h' : tf.get_variable('in_vid_1_layer_2_states_h', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'in_vid_2_layer_1_states_c' : tf.get_variable('in_vid_2_layer_1_states_c', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_2_layer_1_states_h' : tf.get_variable('in_vid_2_layer_1_states_h', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'in_vid_2_layer_2_states_c' : tf.get_variable('in_vid_2_layer_2_states_c', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_2_layer_2_states_h' : tf.get_variable('in_vid_2_layer_2_states_h', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'in_vid_3_layer_1_states_c' : tf.get_variable('in_vid_3_layer_1_states_c', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_3_layer_1_states_h' : tf.get_variable('in_vid_3_layer_1_states_h', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'in_vid_3_layer_2_states_c' : tf.get_variable('in_vid_3_layer_2_states_c', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_3_layer_2_states_h' : tf.get_variable('in_vid_3_layer_2_states_h', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'in_vid_4_layer_1_states_c' : tf.get_variable('in_vid_4_layer_1_states_c', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_4_layer_1_states_h' : tf.get_variable('in_vid_4_layer_1_states_h', shape=(1,num_hidden_1), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'in_vid_4_layer_2_states_c' : tf.get_variable('in_vid_4_layer_2_states_c', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'in_vid_4_layer_2_states_h' : tf.get_variable('in_vid_4_layer_2_states_h', shape=(1,num_hidden_2), 
                                     initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'enc_1_layer_1_states_c' : tf.get_variable('enc_1_layer_1_states_c', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'enc_1_layer_1_states_h' : tf.get_variable('enc_1_layer_1_states_h', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'enc_1_layer_2_states_c' : tf.get_variable('enc_1_layer_2_states_c', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'enc_1_layer_2_states_h' : tf.get_variable('enc_1_layer_2_states_h', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'enc_2_layer_1_states_c' : tf.get_variable('enc_2_layer_1_states_c', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'enc_2_layer_1_states_h' : tf.get_variable('enc_2_layer_1_states_h', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'enc_2_layer_2_states_c' : tf.get_variable('enc_2_layer_2_states_c', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'enc_2_layer_2_states_h' : tf.get_variable('enc_2_layer_2_states_h', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'dec_1_layer_1_states_c' : tf.get_variable('dec_1_layer_1_states_c', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'dec_1_layer_1_states_h' : tf.get_variable('dec_1_layer_1_states_h', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'dec_1_layer_2_states_c' : tf.get_variable('dec_1_layer_2_states_c', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'dec_1_layer_2_states_h' : tf.get_variable('dec_1_layer_2_states_h', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'dec_1_layer_3_states_c' : tf.get_variable('dec_1_layer_3_states_c', shape=(1,self_output_size), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'dec_1_layer_3_states_h' : tf.get_variable('dec_1_layer_3_states_h', shape=(1,self_output_size), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'enc_3_layer_1_states_c' : tf.get_variable('enc_3_layer_1_states_c', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'enc_3_layer_1_states_h' : tf.get_variable('enc_3_layer_1_states_h', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        'enc_3_layer_2_states_c' : tf.get_variable('enc_3_layer_2_states_c', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'enc_3_layer_2_states_h' : tf.get_variable('enc_3_layer_2_states_h', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),

        'dec_2_layer_1_states_c' : tf.get_variable('dec_2_layer_1_states_c', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'dec_2_layer_1_states_h' : tf.get_variable('dec_2_layer_1_states_h', shape=(1,num_hidden_1), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=False),
        
        'dec_2_layer_2_states_c' : tf.get_variable('dec_2_layer_2_states_c', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        'dec_2_layer_2_states_h' : tf.get_variable('dec_2_layer_2_states_h', shape=(1,num_hidden_2), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True),
        
        #'dec_2_layer_3_states_c' : tf.get_variable('dec_2_layer_3_states_c', shape=(1,video_input_size+video_height*video_width), 
        'dec_2_layer_3_states_c' : tf.get_variable('dec_2_layer_3_states_c', shape=(1,video_input_size), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True), 
        #'dec_2_layer_3_states_h' : tf.get_variable('dec_2_layer_3_states_h', shape=(1,video_input_size+video_height*video_width), 
        'dec_2_layer_3_states_h' : tf.get_variable('dec_2_layer_3_states_h', shape=(1,video_input_size), 
                                  initializer = tf.contrib.layers.xavier_initializer(), trainable=True), 
    }
    

    return weights, biases, states

def model_eval(vision_0, vision_minus_1, vision_minus_2, vision_minus_3, 
               vision_minus_4, self_output, last_prediction, last_loss, step_number):
    # the basic model definition.  it receives as input per-pixel information about:
    # the current pixels (vision_0)  -pixel value 'position'
    # the change in current pixels (vision_minus_1)  - this is essentially the dV/dt of vision_0
    #                                                  with V being the pixel values, t being 1 time step
    #                                                  essentially a pixel value 'velocity'
    # d^2V/dt^2 (vision_minus_2) - pixel value 'acceleration'
    # d^3V/dt^3 (vision_minus_3) - pixel value 'jerk'
    # d^4V/dt^4 (vision_minus_4) - pixel value 'jounce'
    #
    # additional, the model is conditioned on (and takes as additional input):
    # its previous set of actions (self_output)
    # its previous loss (last_loss)
    # its previous prediction (last_prediction)
    #
    # step_number is not currently used, but is included as an argument for future experimentation
    
    # I fully realize that using fully connected layers for image processing is non-ideal compared to cnns.
    # this prototype work is aimed to serve as a general proof of concept
    
    # each lstm layer requires a new namespace otherwise the internal machinations will collide
    with tf.variable_scope('v0l1'):
        in_vid_0_layer_1_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_1, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_0_layer_1_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_0_layer_1_states_c'], 
                                        states['in_vid_0_layer_1_states_h'])
        in_vid_0_layer_1_outputs, in_vid_0_layer_1_states = in_vid_0_layer_1_cell(vision_0, 
                                                                                  in_vid_0_layer_1_prev)
        in_vid_0_layer_1 = tf.reshape(in_vid_0_layer_1_outputs, [1, num_hidden_1])
    
    with tf.variable_scope('v0l2'):
        in_vid_0_layer_2_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_2, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_0_layer_2_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_0_layer_2_states_c'], 
                                        states['in_vid_0_layer_2_states_h'])
        in_vid_0_layer_2_outputs, in_vid_0_layer_2_states = in_vid_0_layer_2_cell(in_vid_0_layer_1, 
                                                                                  in_vid_0_layer_2_prev)
        in_vid_0_layer_2 = tf.reshape(in_vid_0_layer_2_outputs, [1, num_hidden_2])
        
    with tf.variable_scope('v1l1'):
        in_vid_1_layer_1_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_1, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_1_layer_1_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_1_layer_1_states_c'], 
                                        states['in_vid_1_layer_1_states_h'])
        in_vid_1_layer_1_outputs, in_vid_1_layer_1_states = in_vid_1_layer_1_cell(vision_minus_1, 
                                                                                  in_vid_1_layer_1_prev)
        in_vid_1_layer_1 = tf.reshape(in_vid_1_layer_1_outputs, [1, num_hidden_1])

    with tf.variable_scope('v1l2'):
        in_vid_1_layer_2_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_2, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_1_layer_2_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_1_layer_2_states_c'], 
                                        states['in_vid_1_layer_2_states_h'])
        in_vid_1_layer_2_outputs, in_vid_1_layer_2_states = in_vid_1_layer_2_cell(in_vid_1_layer_1, 
                                                                                  in_vid_1_layer_2_prev)
        in_vid_1_layer_2 = tf.reshape(in_vid_1_layer_2_outputs, [1, num_hidden_2])
        
    with tf.variable_scope('v2l1'):
        in_vid_2_layer_1_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_1, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_2_layer_1_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_2_layer_1_states_c'], 
                                        states['in_vid_2_layer_1_states_h'])
        in_vid_2_layer_1_outputs, in_vid_2_layer_1_states = in_vid_2_layer_1_cell(vision_minus_2, 
                                                                                  in_vid_2_layer_1_prev)
        in_vid_2_layer_1 = tf.reshape(in_vid_2_layer_1_outputs, [1, num_hidden_1])
    
    with tf.variable_scope('v2l2'):
        in_vid_2_layer_2_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_2, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_2_layer_2_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_2_layer_2_states_c'], 
                                        states['in_vid_2_layer_2_states_h'])
        in_vid_2_layer_2_outputs, in_vid_2_layer_2_states = in_vid_2_layer_2_cell(in_vid_2_layer_1, 
                                                                                  in_vid_2_layer_2_prev)
        in_vid_2_layer_2 = tf.reshape(in_vid_2_layer_2_outputs, [1, num_hidden_2])

    with tf.variable_scope('v3l1'):
        in_vid_3_layer_1_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_1, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_3_layer_1_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_3_layer_1_states_c'], 
                                        states['in_vid_3_layer_1_states_h'])
        in_vid_3_layer_1_outputs, in_vid_3_layer_1_states = in_vid_3_layer_1_cell(vision_minus_3, 
                                                                                  in_vid_3_layer_1_prev)
        in_vid_3_layer_1 = tf.reshape(in_vid_3_layer_1_outputs, [1, num_hidden_1])
    
    with tf.variable_scope('v3l2'):
        in_vid_3_layer_2_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_2, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_3_layer_2_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_3_layer_2_states_c'], 
                                        states['in_vid_3_layer_2_states_h'])
        in_vid_3_layer_2_outputs, in_vid_3_layer_2_states = in_vid_3_layer_2_cell(in_vid_3_layer_1, 
                                                                                  in_vid_3_layer_2_prev)
        in_vid_3_layer_2 = tf.reshape(in_vid_3_layer_2_outputs, [1, num_hidden_2])
 
    with tf.variable_scope('v4l1'):
        in_vid_4_layer_1_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_1, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_4_layer_1_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_4_layer_1_states_c'], 
                                        states['in_vid_4_layer_1_states_h'])
        in_vid_4_layer_1_outputs, in_vid_4_layer_1_states = in_vid_4_layer_1_cell(vision_minus_4, 
                                                                                  in_vid_4_layer_1_prev)
        in_vid_4_layer_1 = tf.reshape(in_vid_4_layer_1_outputs, [1, num_hidden_1])
    
    with tf.variable_scope('v4l2'):
        in_vid_4_layer_2_cell = tf.contrib.rnn.LSTMCell(
                                    num_hidden_2, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        in_vid_4_layer_2_prev = tf.contrib.rnn.LSTMStateTuple(
                                        states['in_vid_4_layer_2_states_c'], 
                                        states['in_vid_4_layer_2_states_h'])
        in_vid_4_layer_2_outputs, in_vid_4_layer_2_states = in_vid_1_layer_2_cell(in_vid_4_layer_1, 
                                                                                  in_vid_4_layer_2_prev)
        in_vid_4_layer_2 = tf.reshape(in_vid_4_layer_2_outputs, [1, num_hidden_2])
    
    
    enc_1_layer_1 = tf.nn.sigmoid(
                        tf.add(
                            tf.matmul(
                                tf.concat(
                                    [
                                     in_vid_4_layer_2, 
                                     in_vid_3_layer_2, 
                                     in_vid_2_layer_2, 
                                     in_vid_1_layer_2, 
                                     in_vid_0_layer_2
                                    ], 
                                1),
                            weights['enc_1_h1']), 
                        biases['enc_1_b1'])
                    )
    
    enc_1_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(enc_1_layer_1, weights['enc_1_h2']), biases['enc_1_b2']))
    
    output_feedback = tf.concat([self_output, last_prediction, last_loss], 1)
    
    enc_2_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(output_feedback, weights['enc_2_h1']), biases['enc_2_b1']))
    
    enc_2_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(enc_2_layer_1, weights['enc_2_h2']), biases['enc_2_b2']))
    
    dec_1_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.concat([enc_2_layer_2, enc_1_layer_2], 1),
                                                            weights['dec_1_h1']), biases['dec_1_b1']))
    
    dec_1_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(dec_1_layer_1, weights['dec_1_h2']), biases['dec_1_b2']))
    
    dec_1_layer_3_pre = tf.add(tf.matmul(dec_1_layer_2, weights['dec_1_h3']), biases['dec_1_b3'])
    
    # splitting output data 
    # this one gotta be taneh! 
    # (spelling errors intentional to preserve meaning through 'find and replace' of activation functions)
    updown_leftright = tf.multiply(tf.nn.tanh(tf.slice(dec_1_layer_3_pre, [0, 0], [1, 2])), 2.0)
    # and this one gotta be sigmoied
    req_data = tf.nn.sigmoid(tf.slice(dec_1_layer_3_pre, [0, 2], [1, 13]))
    req_data_max = tf.argmax(req_data, axis=1)
    
    # and recombining  # addition of prev act doesnt take place here, its in _pre
    dec_1_layer_3 = tf.concat([updown_leftright, req_data], 1)
    
    enc_3_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(dec_1_layer_3, weights['enc_3_h1']), biases['enc_3_b1']))
    
    enc_3_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(enc_3_layer_1, weights['enc_3_h2']), biases['enc_3_b2']))
    
    dec_2_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.concat([enc_3_layer_2, enc_1_layer_2], 1),
                                                            weights['dec_2_h1']), biases['dec_2_b1']))
    
    dec_2_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(dec_2_layer_1, weights['dec_2_h2']), biases['dec_2_b2']))
    
    # might need to not add in the prev act on this last prediction layer. dunno.
    
    dec_2_layer_3 = tf.nn.sigmoid(
                        tf.add(
                            tf.matmul(
                                dec_2_layer_2, 
                                weights['dec_2_h3']
                            ), 
                            biases['dec_2_b3']
                        )
                    )
    prediction = dec_2_layer_3 
    output_data = dec_1_layer_3
    
    updown_leftright_data = updown_leftright

    # tf doesnt have stateful lstm built in, so....
    tf.assign(states['in_vid_0_layer_1_states_c'], in_vid_0_layer_1_states[0])
    tf.assign(states['in_vid_0_layer_1_states_h'], in_vid_0_layer_1_states[1])
    tf.assign(states['in_vid_0_layer_2_states_c'], in_vid_0_layer_2_states[0])
    tf.assign(states['in_vid_0_layer_2_states_h'], in_vid_0_layer_2_states[1])
    tf.assign(states['in_vid_1_layer_1_states_c'], in_vid_1_layer_1_states[0])
    tf.assign(states['in_vid_1_layer_1_states_h'], in_vid_1_layer_1_states[1])
    tf.assign(states['in_vid_1_layer_2_states_c'], in_vid_1_layer_2_states[0])
    tf.assign(states['in_vid_1_layer_2_states_h'], in_vid_1_layer_2_states[1])
    tf.assign(states['in_vid_2_layer_1_states_c'], in_vid_2_layer_1_states[0])
    tf.assign(states['in_vid_2_layer_1_states_h'], in_vid_2_layer_1_states[1])
    tf.assign(states['in_vid_2_layer_2_states_c'], in_vid_2_layer_2_states[0])
    tf.assign(states['in_vid_2_layer_2_states_h'], in_vid_2_layer_2_states[1])
    tf.assign(states['in_vid_3_layer_1_states_c'], in_vid_3_layer_1_states[0])
    tf.assign(states['in_vid_3_layer_1_states_h'], in_vid_3_layer_1_states[1])
    tf.assign(states['in_vid_3_layer_2_states_c'], in_vid_3_layer_2_states[0])
    tf.assign(states['in_vid_3_layer_2_states_h'], in_vid_3_layer_2_states[1])
    tf.assign(states['in_vid_4_layer_1_states_c'], in_vid_4_layer_1_states[0])
    tf.assign(states['in_vid_4_layer_1_states_h'], in_vid_4_layer_1_states[1])
    tf.assign(states['in_vid_4_layer_2_states_c'], in_vid_4_layer_2_states[0])
    tf.assign(states['in_vid_4_layer_2_states_h'], in_vid_4_layer_2_states[1])
    
    return prediction, output_data, updown_leftright_data, req_data_max

# defining the possible directional requests (to user)
request_images = {0: np.array(cv2.imread('./reqs/back.bmp')),
                 1: np.array(cv2.imread('./reqs/down.bmp')),
                 2: np.array(cv2.imread('./reqs/forward.bmp')),
                 3: np.array(cv2.imread('./reqs/left.bmp')),
                 4: np.array(cv2.imread('./reqs/right.bmp')),
                 5: np.array(cv2.imread('./reqs/rot_ccw.bmp')),
                 6: np.array(cv2.imread('./reqs/rot_cw.bmp')),
                 7: np.array(cv2.imread('./reqs/rot_down.bmp')),
                 8: np.array(cv2.imread('./reqs/rot_left.bmp')),
                 9: np.array(cv2.imread('./reqs/rot_right.bmp')),
                 10: np.array(cv2.imread('./reqs/rot_up.bmp')),
                 11: np.array(cv2.imread('./reqs/stay.bmp')),
                 12: np.array(cv2.imread('./reqs/up.bmp'))}

def loss_fxn(prediction, vision_target, vision_0, step_number):
    # loss function calculation
    
    # the goal is to balance the loss between the model trying to accurately predict the future
    # and trying to 'learn new things'
    #
    # this is fundamentally challenging because those two objectives are essentially in exact opposition 
    # to each other.
    #
    # initial experimentation with the 'learning new things' part is based on penalizing high correlation
    # values between the input vision frame and the (future) objective vision frame.  
    #
    # this does indeed result in the model occasionally 'getting bored' once its predictions start getting 
    # better, and it starts 'looking around' a bit.
    #
    # i dont like this implementation though, and i'd prefer to base the 'learn new things' objective on
    # something like the magnitude of the change in network weights during the backpropagation pass.
    # but i dont know how to extract those deltas from tensorflow yet.
    
    pixels = prediction
    #pixels = tf.slice(prediction, [0, 0], [1, video_input_size])
    #confidence = tf.slice(prediction, [0, video_input_size], [1, video_height*video_width])
    
    with tf.name_scope('losses'):
        with tf.name_scope('pred_error'):
            l1 = tf.abs(tf.subtract(vision_target/255.0, pixels))
            l2 = tf.pow(tf.subtract(vision_target/255.0, pixels),2.0)
            
            # this current formulation of the loss function combines l1 and l2 losses such that at large 
            # error values, the loss gradient is constant, but at small errors, it is diminishing.
            pred_error = tf.reshape(
                            tf.reduce_mean(
                                tf.subtract(
                                    tf.add(
                                        tf.minimum(
                                            l2,
                                            tf.constant([0.5])
                                        ), 
                                        tf.maximum(
                                            l1, 
                                            tf.constant([0.5])
                                        )
                                    ), 
                                    tf.constant([0.5])
                                )
                            ) 
                            ,
                            []
                        )
            
            tf.summary.scalar('pred_error', pred_error)
        with tf.name_scope('vis_error'):
            vis_error = tf.abs(tf.subtract(vision_target/255.0, vision_0/255.0))
            tf.summary.scalar('vis_error', tf.reduce_mean(vis_error))
        with tf.name_scope('pred_minus_vis'):
            diff_error = tf.reduce_mean(pred_error-vis_error)
            tf.summary.scalar('pred_minus_vis', diff_error)
            #x = confidence
            #y = rse
            #z = tf.maximum(-1.0*tf.pow((-y-x+1),3),0)
            #conf_loss = tf.reduce_mean(z)
            #tf.summary.scalar('confidence_fxn_loss', conf_loss)
            
        noise = tf.reduce_sum(tf.random_uniform((1, 1), minval=-0.002, maxval=0.002, dtype=tf.float32))
        
        x_mean = tf.reduce_mean(vision_target)
        y_mean = tf.reduce_mean(vision_0)
        x_diff = tf.subtract(tf.reshape(vision_target, [-1]), x_mean)
        y_diff = tf.subtract(tf.reshape(vision_0, [-1]), y_mean)

        pearson_correlation = tf.divide(
                                  tf.tensordot(
                                      x_diff, 
                                      y_diff, 
                                      1
                                  ),
                                  tf.multiply(
                                      tf.sqrt(
                                          tf.reduce_sum(
                                              tf.square(
                                                  x_diff
                                              )
                                          )
                                      ),
                                      tf.sqrt(
                                          tf.reduce_sum(
                                              tf.square(
                                                  y_diff
                                              )
                                          )
                                      )
                                  )
                            )
        
        with tf.name_scope('correlation_loss'):
            corr_loss = tf.abs(pearson_correlation)
            tf.summary.scalar('corr_loss', corr_loss)
        
        loss = (pred_error + corr_loss) * (1.0+0.0) #noise)
        
    return loss, l1
    #return pred_error, l1
    

def get_vid_frame(cap):
    # python generator because its the only/best way to feed live, streaming data into a tf graph ive found
    while True:
        # second while loop here is to help curb 'session timeout' errors.  
        # they still happen sometimes though, so be sure to checkpoint
        ret = False
        while ret==False:
            ret, frame = cap.read()
        if video_channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        visual = np.array(frame).reshape(full_video_height, full_video_width, video_channels).astype(np.float32)
        
        # restrict the space of perceivable colors to exclude pure white/black in order to prevent
        #   bleaching out of receptive fields.
        visual *= 0.8
        visual += 255*0.1
        ret=False
        yield visual
        
    
def reduce_vision_window(visual_data, input_x_coord, input_y_coord, updown_leftright):
    # function to manipulate the vision input frame
    updown = tf.reshape(tf.slice(updown_leftright, [0,0], [1,1]), [])
    leftright = tf.reshape(tf.slice(updown_leftright, [0,1], [1,1]), [])
    
    x_diff = tf.cast(tf.minimum(tf.maximum(input_x_coord + tf.cast(with_selfmovement*leftright, tf.int32), 0), full_video_width-video_width-1) - input_x_coord, tf.float32)
    y_diff = tf.cast(tf.minimum(tf.maximum(input_y_coord + tf.cast(with_selfmovement*updown, tf.int32), 0), full_video_height-video_height-1) - input_y_coord, tf.float32)
    
    # micro-saccade implementation (small random motions that occur especially when vision is stationary)
    micsac = tf.random_uniform([2], minval = -micsac_magnitude, maxval = micsac_magnitude)
    delta_y = tf.cast(tf.add(tf.multiply(micsac[0], tf.exp(tf.divide(tf.abs(y_diff), -2.0))), y_diff), tf.int32)
    delta_x = tf.cast(tf.add(tf.multiply(micsac[1], tf.exp(tf.divide(tf.abs(x_diff), -2.0))), x_diff), tf.int32)
    ###
    
    new_x_coord = tf.minimum(tf.maximum(input_x_coord + delta_x, 0), full_video_width-video_width-1)
    new_y_coord = tf.minimum(tf.maximum(input_y_coord + delta_y, 0), full_video_height-video_height-1)
    
    new_coords = tf.reshape(tf.concat([new_y_coord, new_x_coord, [0]], 0), [3])
    
    visual_small_window = tf.slice(visual_data, new_coords, [video_height, video_width, video_channels])
    
    visual_small = tf.cast(tf.reshape(visual_small_window, [1,video_height*video_width*video_channels]), tf.float32)
        
    return visual_small, new_x_coord, new_y_coord
        
    
weights, biases, states = get_weights_biases()

def main():
    # start video feed
    
    cap = cv2.VideoCapture(webcam_1_index)
    cap.set(3,full_video_height)
    cap.set(4,full_video_width)   
    
    current_x_coord = np.array([int((full_video_width-video_width)/2.0)])
    current_y_coord = np.array([int((full_video_width-video_width)/2.0)])
    
    with tf.Session() as sess:
        
        last_loss, last_prediction, step_number, vision_lag, vision_target, vision_0, vision_minus_1, vision_minus_2, vision_minus_3, vision_minus_4, self_output, input_x_coord, input_y_coord, updown_leftright, visual_data = placeholder_inputs()
        
        # create and fill vision queue.  I'm using queues because it seemed like the best structure
        # for manipulating sequential data input.  though the m1/m2/m3 queues could probably just be variables
        vision_target_queue = tf.FIFOQueue(capacity=6+num_video_lag_steps, 
                                       dtypes=[tf.float32], 
                                       shapes=[1, video_input_size], 
                                       name='vision_target_q')
        
        vision_lag_queue = tf.FIFOQueue(capacity=1+num_video_lag_steps, 
                                       dtypes=[tf.float32], 
                                       shapes=[1, video_input_size], 
                                       name='vision_lag_q')
        
        vision_0_queue = tf.FIFOQueue(capacity=1, 
                                      dtypes=[tf.float32], 
                                      shapes=[1, video_input_size], 
                                      name='vision_0_q')
        
        vision_m1_queue = tf.FIFOQueue(capacity=1, 
                                      dtypes=[tf.float32], 
                                      shapes=[1, video_input_size], 
                                      name='vision_m1_q')
        
        vision_m2_queue = tf.FIFOQueue(capacity=1, 
                                      dtypes=[tf.float32], 
                                      shapes=[1, video_input_size], 
                                      name='vision_m2_q')
        
        vision_m3_queue = tf.FIFOQueue(capacity=1, 
                                      dtypes=[tf.float32], 
                                      shapes=[1, video_input_size], 
                                      name='vision_m3_q')
        
        vision_m4_queue = tf.FIFOQueue(capacity=1, 
                                      dtypes=[tf.float32], 
                                      shapes=[1, video_input_size], 
                                      name='vision_m4_q')
        
        # we need to grab video frames via a python generator because of tensorflow reasons.
        v_target_data = get_vid_frame(cap)
        v_target_dataset = tf.data.Dataset.from_generator(lambda: v_target_data, tf.float32)
        vision_target_frame = v_target_dataset.make_one_shot_iterator().get_next()
        
        prediction, output_data, updown_leftright_data, req_data_max = model_eval(vision_0, 
                                                                         vision_minus_1, 
                                                                         vision_minus_2, 
                                                                         vision_minus_3, 
                                                                         vision_minus_4, 
                                                                         self_output, 
                                                                         last_prediction, 
                                                                         last_loss,
                                                                         step_number)
        
        visual_small_op = reduce_vision_window(visual_data, input_x_coord, input_y_coord, updown_leftright)
        
        
        v_target_enqueue_op = vision_target_queue.enqueue(vision_target)
        v_lag_enqueue_op = vision_lag_queue.enqueue(vision_lag)
        v_0_enqueue_op = vision_0_queue.enqueue(vision_0)
        v_m1_enqueue_op = vision_m1_queue.enqueue(vision_minus_1)
        v_m2_enqueue_op = vision_m2_queue.enqueue(vision_minus_2)
        v_m3_enqueue_op = vision_m3_queue.enqueue(vision_minus_3)
        v_m4_enqueue_op = vision_m4_queue.enqueue(vision_minus_4)
        
        
        v_target_dequeue_op = vision_target_queue.dequeue()
        v_lag_dequeue_op = vision_lag_queue.dequeue()
        v_0_dequeue_op = vision_0_queue.dequeue()
        v_m1_dequeue_op = vision_m1_queue.dequeue()
        v_m2_dequeue_op = vision_m2_queue.dequeue()
        v_m3_dequeue_op = vision_m3_queue.dequeue()
        v_m4_dequeue_op = vision_m4_queue.dequeue()
        
        # initializing the input queues
        for i in range(6+num_video_lag_steps):
            visual_data_out = sess.run(vision_target_frame)
            visual_small_data, new_x_coord, new_y_coord = sess.run(visual_small_op, feed_dict={visual_data: visual_data_out,
                                                                input_x_coord: current_x_coord,
                                                                input_y_coord: current_y_coord,
                                                                updown_leftright: np.array([[0,0]])})
            current_x_coord = new_x_coord
            current_y_coord = new_y_coord
            #visual_enq_data = np.concatenate((visual_small_data[0], current_x_coord, current_y_coord))
            visual_enq_data = visual_small_data[0]
            
            sess.run(v_target_enqueue_op, feed_dict={vision_target: [visual_enq_data]})
        
        # init the queues.
        
        v_t_m4_pos = sess.run(v_target_dequeue_op)
        v_t_m3_pos = sess.run(v_target_dequeue_op)
        v_t_m2_pos = sess.run(v_target_dequeue_op)
        v_t_m1_pos = sess.run(v_target_dequeue_op)
        v_t_0_pos = sess.run(v_target_dequeue_op)
        
        for i in range(num_video_lag_steps):
            v_t_lag_pos = sess.run(v_target_dequeue_op)
            sess.run(v_lag_enqueue_op, feed_dict={vision_lag: v_t_lag_pos})
        
        sess.run(v_m4_enqueue_op, feed_dict={vision_minus_4: v_t_m4_pos})
        sess.run(v_m3_enqueue_op, feed_dict={vision_minus_3: v_t_m3_pos})
        sess.run(v_m2_enqueue_op, feed_dict={vision_minus_2: v_t_m2_pos})
        sess.run(v_m1_enqueue_op, feed_dict={vision_minus_1: v_t_m1_pos})
        sess.run(v_0_enqueue_op, feed_dict={vision_0: v_t_0_pos})
        
        # init done.
        
        # create and fill self_output_queue
        self_output_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.float32], shapes=[1, self_output_size], name='self_vision_q')
        so_enqueue_op = self_output_queue.enqueue(self_output)
        so_dequeue_op = self_output_queue.dequeue()
        
        sess.run(
            self_output_queue.enqueue(
                np.reshape(
                    np.append(
                        np.zeros(
                            (1, 
                             self_output_size-2
                            )
                        ), 
                        [[current_x_coord, 
                        current_y_coord
                        ]]
                    ),
                    (1, 
                     self_output_size
                    )
                )
            )
        )
        
        ###########
        # create and fill last_loss queue
        last_loss_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.float32], shapes=[1, video_input_size], name='last_loss_q')
        ll_enqueue_op = last_loss_queue.enqueue(last_loss)
        ll_dequeue_op = last_loss_queue.dequeue()
        
        sess.run(last_loss_queue.enqueue(np.ones((1, video_input_size))))
        
        ###########
        # create and fill last_prediction queue
        last_prediction_queue = tf.FIFOQueue(capacity=1, dtypes=[tf.float32], shapes=[1, video_input_size], name='last_prediction_q')
        lp_enqueue_op = last_prediction_queue.enqueue(last_prediction)
        lp_dequeue_op = last_prediction_queue.dequeue()
        
        sess.run(last_prediction_queue.enqueue(np.zeros((1, video_input_size))))
        
        ###########
                 
        loss, loss_pixels = loss_fxn(prediction, vision_target, vision_0, step_number)
        with tf.name_scope('total_loss'):
            tf.summary.scalar('total_loss', loss)
        
        # i havent used global_step well in this code yet.
        global_step = tf.Variable(0, name='global_step', trainable=False)
    
        # commented-out code below is a basic attempt to grab the backprop weight deltas.  
        # but it operates far too slowly to be feasible.
        #
        # works by making a copy of all the variables and calculating the deltas at each step of training.
        
        
        #joined_vars = tf.concat([tf.reshape(x, [-1]) for x in tf.trainable_variables()], axis=0)
        
        #joined_backup = tf.Variable(np.zeros([int(x) for x in joined_vars.get_shape()]),
        #                            dtype=joined_vars.dtype,
        #                            trainable=False)
        
        #make_backup = tf.assign(joined_backup, joined_vars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate, adam_beta1, adam_beta2)
        #grads_and_vars = optimizer.compute_gradients(loss)
        #grad clipping
        grads, variables = zip(*optimizer.compute_gradients(loss))
        grads = [
            None if gradient is None else tf.clip_by_norm(gradient, 0.0001)
            for gradient in grads]
        
        train_op = optimizer.apply_gradients(zip(grads, variables))
        #train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        #with tf.control_dependencies([make_backup]):
        #    optimizer = tf.train.AdamOptimizer()
        #    train_op = optimizer.minimize(loss, global_step=global_step)
        #
        #with tf.control_dependencies([train_op]):
        #    new_joined = tf.concat([tf.reshape(x, [-1]) for x in tf.trainable_variables()], axis=0)
        #    delta = new_joined - joined_backup
        #    total_sq_delta = tf.reduce_sum(tf.square(delta))
            
            
        summary = tf.summary.merge_all()
        
        summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        init = tf.global_variables_initializer()
        
        sess.run(init)
        #print(grads_and_vars)
        saver = tf.train.Saver()
        
        prev_time = time.time()
        i = 0.0
        while True:
            i += 1.0
            # main training loop here
            
            
            # get inputs to model iteration
            # instead of feeding raw image data, we will calculate nth derivatives of the pixel values.
            v_t_target_pos = sess.run(v_target_dequeue_op)
            
            sess.run(v_lag_enqueue_op, feed_dict={vision_lag: v_t_target_pos})
            v_t_lag_pos = sess.run(v_lag_dequeue_op)
            
            v_t_0_pos = sess.run(v_0_dequeue_op)
            v_t_m1_pos = sess.run(v_m1_dequeue_op)
            v_t_m2_pos = sess.run(v_m2_dequeue_op)
            v_t_m3_pos = sess.run(v_m3_dequeue_op)
            v_t_m4_pos = sess.run(v_m4_dequeue_op)
            
            # calc derivs.  this is currently using an assumption that each training step is 1 time step
            # but i want to include the actual time values in order to get more accurate derivatives.
        
            v_t_0_velo = v_t_0_pos - v_t_m1_pos
            v_t_m1_velo = v_t_m1_pos - v_t_m2_pos
            v_t_m2_velo = v_t_m2_pos - v_t_m3_pos
            v_t_m3_velo = v_t_m3_pos - v_t_m4_pos

            v_t_0_accel = v_t_0_velo- v_t_m1_velo
            v_t_m1_accel = v_t_m1_velo- v_t_m2_velo
            v_t_m2_accel = v_t_m2_velo- v_t_m3_velo

            v_t_0_jerk = v_t_0_accel- v_t_m1_accel
            v_t_m1_jerk = v_t_m1_accel- v_t_m2_accel

            v_t_0_jounce = v_t_0_jerk- v_t_m1_jerk
            
            queued_output_data = sess.run(so_dequeue_op)
            
            last_loss_pixels_data = sess.run(ll_dequeue_op)
            last_prediction_data = sess.run(lp_dequeue_op)
            
            # the commented out bits that split prediction data are alternative lines that allow for
            # last_prediction_data to include both the actual last prediction, as well as a video input-sized
            # 'confidence' matrix.  thats currently just experimental though. obviously slows down computations
            
            last_prediction_err = v_t_m1_pos - last_prediction_data
            #last_prediction_err = v_t_m1_pos - last_prediction_data[0:1,0:video_input_size]
            
            
            #run model
            (_, prediction_out, output_data_out, 
                 updown_leftright_data_out, 
                     req_data_max_out, loss_out, loss_pixels_out, 
                         summary_out) = sess.run([train_op, 
                                                          prediction, 
                                                          output_data, 
                                                          updown_leftright_data,
                                                          req_data_max,
                                                          loss, 
                                                            loss_pixels,
                                                            summary,],
                                                      feed_dict={vision_target: v_t_target_pos,
                                                                 vision_0: v_t_0_pos, 
                                                                 vision_minus_1: v_t_0_velo, 
                                                                 vision_minus_2: v_t_0_accel, 
                                                                 vision_minus_3: v_t_0_jerk, 
                                                                 vision_minus_4: v_t_0_jounce, 
                                                                 self_output:queued_output_data,
                                                                 last_prediction: last_prediction_err,
                                                                 last_loss: last_loss_pixels_data,
                                                                 step_number: [i]})
            
            summary_writer.add_summary(summary_out, i)
            summary_writer.flush()
            
            pred_pixels = prediction_out
            #pred_pixels = prediction_out[0:1,0:video_input_size]
            #conf_pixels = np.multiply(prediction_out[0:1,video_input_size:video_height*video_width+video_input_size], np.zeros((1, video_input_size))+1)
            
            
            # update queues for next iteration.
            visual_data_out = sess.run(vision_target_frame)
            visual_small_data, new_x_coord, new_y_coord = sess.run(visual_small_op, feed_dict={visual_data: visual_data_out,
                                                                input_x_coord: current_x_coord,
                                                                input_y_coord: current_y_coord,
                                                                updown_leftright: updown_leftright_data_out})
            
            
            current_x_coord = new_x_coord
            current_y_coord = new_y_coord
            
            so_enq_data = np.reshape(np.append(np.array(output_data_out), [current_x_coord, current_y_coord]), (1,self_output_size))
            visual_enq_data = visual_small_data[0]
            
            sess.run(v_target_enqueue_op, feed_dict={vision_target: [visual_enq_data]})
            
            sess.run(v_0_enqueue_op, feed_dict={vision_0: v_t_lag_pos})
            sess.run(v_m1_enqueue_op, feed_dict={vision_minus_1: v_t_0_pos})
            sess.run(v_m2_enqueue_op, feed_dict={vision_minus_2: v_t_m1_pos})
            sess.run(v_m3_enqueue_op, feed_dict={vision_minus_3: v_t_m2_pos})
            sess.run(v_m4_enqueue_op, feed_dict={vision_minus_4: v_t_m3_pos})
            sess.run(so_enqueue_op, feed_dict={self_output: so_enq_data})
            
            sess.run(ll_enqueue_op, feed_dict={last_loss: loss_pixels_out})
            sess.run(lp_enqueue_op, feed_dict={last_prediction: pred_pixels})
            
            # display windows
            this_frame = np.resize(v_t_0_pos,(video_height, video_width, video_channels)).astype(np.uint8)
            pred_frame = np.resize(255.0*pred_pixels,(video_height, video_width, video_channels)).astype(np.uint8)
            loss_frame = np.resize(255.0*loss_pixels_out, (video_height, video_width, video_channels)).astype(np.uint8)
            #conf_frame = np.resize(255.0*conf_pixels,(video_height, video_width, video_channels)).astype(np.uint8)
            actu_frame = np.resize(v_t_target_pos,(video_height, video_width, video_channels)).astype(np.uint8)
            
            #print(req_data_max_out)
            req_frame = request_images[req_data_max_out[0]]
            
            cv2.imshow('vision', this_frame)    
            cv2.imshow('request', req_frame)
            cv2.imshow('prediction', pred_frame)
            cv2.imshow('loss', loss_frame)
            #cv2.imshow('confidence', conf_frame)
            cv2.imshow('actual', actu_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display logs per step
            if i % display_step == 0:
                this_time = time.time()
                fps = 100.0/(this_time-prev_time)
                prev_time = time.time()
                print('Step %i: FPS: %i Loss: %f' % (i, int(fps), loss_out))
                #print(self_frame[0][:5])

            # for restore check https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/save_restore_model.py
            #if i % 10000 == 0:
            #    saver.save(sess, '/home/miej/projects/4/backup/model_%d.ckpt' % i)
                
            # for tensorboard, in term run 
            # $ tensorboard --logdir=/tmp/tensorflow/mnist/log
                
main()