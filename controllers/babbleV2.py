### Motor Babbling for NTC Test Bench ###

import sys
sys.path.append("../catkin_ws/devel/lib/python3/dist-packages")

import rospy
from bench.msg import BenchState, BenchMotorControl, BenchRecorderControl
import csv
import time
import random
from matplotlib import pyplot as plt
import numpy as np

def babbling_fcn(babbling_seconds=300, kinematics_activations_show=True):

	#local variables
	timestep=0.01
	run_samples=int(np.round(babbling_seconds/timestep))
	pass_chance = timestep
	skip_rows = 200

	#TODO: Normalizing current values between 0,1. Might have to change that so that data is normalized in ANN implementation
	max_in = 1
	min_in = 0

	#Generating random activations in range (min_in, max_in)
	motor1_act = systemID_input_gen_fcn(babbling_seconds, pass_chance, max_in, min_in, timestep)
	motor2_act = systemID_input_gen_fcn(babbling_seconds, pass_chance, max_in, min_in, timestep)
 	
	babbling_activations = np.transpose(np.concatenate([[motor1_act],[motor2_act]], axis=0))
 
	#Save resulting kinematics data
	if kinematics_activations_show:
		kinematics_activations_show_fcn(activations=babbling_activations)
	[babbling_kinematics, babbling_activations, chassis_pos] = run_activations_fcn(babbling_activations, timestep)
 
	#return kinematics,activations map and skip skip_rows rows (setup phase)
	return babbling_kinematics[skip_rows:,:], babbling_activations[skip_rows:,:]

def systemID_input_gen_fcn(signal_duration_in_seconds, pass_chance, max_in, min_in, timestep):

	number_of_samples = int(np.round(signal_duration_in_seconds/timestep))
	samples = np.linspace(0, signal_duration_in_seconds, number_of_samples)
 
	gen_input = np.zeros(number_of_samples,) * min_in

	#generating number_of_samples random activations
	for i in range(1, number_of_samples):
		pass_rand = np.random.uniform(0,1,1)
	
		if pass_rand < pass_chance:
			gen_input[i] = ((max_in-min_in)*np.random.uniform(0,1,1)) + min_in
		else:
			gen_input[i] = gen_input[i-1]

	return gen_input


def kinematics_activations_show_fcn(vs_time=False, timestep=0.01, **kwargs):

	#plotting the resulting kinematics or activations
	sample_no_kinematics=0
	sample_no_activations=0

	#setting up variables
	if ("kinematics" in kwargs):
		kinematics = kwargs["kinematics"]
		sample_no_kinematics = kinematics.shape[0]

	if ("activations" in kwargs):
		activations = kwargs["activations"]
		sample_no_activations = activations.shape[0]

	if not (("kinematics" in kwargs) or ("activations" in kwargs)):
		raise NameError('Either kinematics or activations needs to be provided')
	
	if (sample_no_kinematics!=0) & (sample_no_activations!=0) & (sample_no_kinematics!=sample_no_activations):
		raise ValueError('Number of samples for both kinematics and activation matrices should be equal and not zero')
	
	else:
		number_of_samples = np.max([sample_no_kinematics, sample_no_activations])
		if vs_time:
			x = np.linspace(0,timestep*number_of_samples,number_of_samples)
		else:
			x = range(number_of_samples)
	 
	#plotting kinematics: angle degree, angular velocity, angular acceleration
	if ("kinematics" in kwargs):
		plt.figure()
	
		plt.subplot(3, 1, 1)
		plt.plot(x, kinematics[:,0])
		plt.ylabel('q0 (rads)')
	
		plt.subplot(3, 1, 2)
		plt.plot(x, kinematics[:,1])
		plt.ylabel('q0 dot (rads/s)')
	
		plt.subplot(3, 1, 3)
		plt.plot(x, kinematics[:,2])
		plt.ylabel('q0 double dot (rads/s^2)')
	
		plt.xlabel('motor 1 activation values')
	
	#plotting activation values
	if ("activations" in kwargs):
		plt.figure()
	
		plt.subplot(2, 1, 1)
		plt.plot(x, activations[:,0])
		plt.ylabel('motor 1 activation values')
	
		plt.subplot(2, 1, 2)
		plt.plot(x, activations[:,1])
		plt.ylabel('motor 2 activation values')

	plt.show(block=True)
 
def run_activations_fcn(activations, timestep):
    return 0;
    #Todo