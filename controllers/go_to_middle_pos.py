import sys
sys.path.append("../catkin_ws/devel/lib/python3/dist-packages")

import rospy
from bench.msg import BenchState, BenchMotorControl, BenchRecorderControl
# Import the required modules
import csv
import time

TOP_ANGLE_VALUE = 1737.0
BOT_ANGLE_VALUE = 1223.0

rospy.init_node('go_to_middle_controller')


bench_control_pub = rospy.Publisher('/test_bench/BenchMotorControl', BenchMotorControl, queue_size=10)

# Define the callback function that will be called
# whenever a message is received on the topic
def callback(bench_state):
    middle_pos = (TOP_ANGLE_VALUE + BOT_ANGLE_VALUE) / 2


    if bench_state.angle < middle_pos:
        direction = 1
    else:
        direction = 0

    if direction == 1:
        msg = BenchMotorControl()
        msg.flex_myobrick_pwm = 7
        msg.extend_myobrick_pwm = -2
        bench_control_pub.publish(msg)


    if direction == 0:
        msg = BenchMotorControl()
        msg.flex_myobrick_pwm = -2
        msg.extend_myobrick_pwm = 7
        bench_control_pub.publish(msg)

    if bench_state.safety_switch_pressed == True:
        print('Kill switch is pressed, stopping.')
        rospy.signal_shutdown('')
        sys.exit()

    if bench_state.flex_myobrick_in_running_state == False:
        print('Flex MyoBrick is not in running state, stopping.')
        rospy.signal_shutdown('')
        sys.exit()

    if bench_state.extend_myobrick_in_running_state == False:
        print('Extend MyoBrick is not in running state, stopping.')
        rospy.signal_shutdown('')
        sys.exit()

    if abs(bench_state.angle - middle_pos) < (TOP_ANGLE_VALUE - BOT_ANGLE_VALUE) / 20 :
        print('Is at center.')
        msg = BenchMotorControl()
        msg.flex_myobrick_pwm = 0
        msg.extend_myobrick_pwm = 0
        bench_control_pub.publish(msg)
        rospy.signal_shutdown('')
        sys.exit()


bench_control_sub = rospy.Subscriber('/test_bench/BenchState', BenchState, callback)

rospy.spin()