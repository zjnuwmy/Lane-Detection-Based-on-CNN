
import numpy as np
import cv2
import pandas as pd
#from ..model import Model

from functions import raw_to_cnn, cnn_to_raw, raw_motor_to_rgb
from img_serializer import deserialize_image

from model import Model
#from datasets import get_servo_dataset

# Visualize images
# With and without any predictions
#DATA_FILES = ['C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_0.csv', 'C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_1.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_2.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_3.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_4.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_5.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_6.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_7.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_8.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_9.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_10.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_11.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_12.csv','C:/Users/circle/Desktop/RCDATA_CSV_7_15/output_13.csv']

def visualize_data(filename, width=70, height=40, depth=3, cnn_model=Model, conf='./settings.json'):
    """
    When cnn_model is specified it'll show what the cnn_model predicts (red)
    as opposed to what inputs it actually received (green)
    """
    #data = pd.read_csv(filename, engine='python', error_bad_lines=False)     
   
    data = pd.read_csv(filename)     
 
    model= Model()
  
   
    for i in data.index:
        cur_img = data['image'][i]
        cur_throttle = float(data['servo'][i])
        cur_motor = float(data['motor'][i])        
        print(cur_motor)
        # [1:-1] is used to remove '[' and ']' from string 
        cur_img_array = deserialize_image(cur_img, config=conf)        
        y_input = cur_img_array.copy() # NN input

        # And then rescale it so we can more easily preview it
        cur_img_array = cv2.resize(cur_img_array, (480, 320), interpolation=cv2.INTER_CUBIC)
        #model.predict(cur_img_array)
        #pred_servo = model.predict(cur_img_array)
        # Extra debugging info (e.g. steering etc)
        cv2.putText(cur_img_array, "frame: %s" % str(i), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cur_throttle=int(1800*(cur_throttle-0.15)+90)    #为什么取0.15作为均值
        cv2.line(cur_img_array, (240, 300), (240-(90-cur_throttle), 200), (0, 255, 0), 3)

        #cv2.line(cur_img_array, (240, 300), (240-(90-cur_throttle), 200), (0, 0, 255), 3)

        # Motor values
        # RGB
        cur_motor=int(1800*(cur_motor-0.15)+90)
        cv2.line(cur_img_array, (50, 160), (50, 160-(90-cur_motor)), raw_motor_to_rgb(cur_motor), 3)

        # If we wanna visualize our cnn_model
        if cnn_model:
            #y = cnn_model.predict([pre])
            y = model.predict(y_input)    #y是float类型的值
            #print(type(y))
            #print(y)
            #servo_out = cnn_to_raw(y)      #这里我想知道 cnn_output与raw_output的关系！！！       
            #print("servo_out:",servo_out)
            print("y_angle:",240-(90-int(1800*(y-0.15)+90)))
            print("servo_out:{} <---> {}".format(data['servo'][i],y))
            cv2.line(cur_img_array, (240, 300), (240-(90-int(1800*(y-0.15)+90)), 200), (0, 0, 255), 3)
            #cv2.line(cur_img_array, (240, 300), (240-(90-int(servo_out)), 200), (0, 0, 255), 3)
            #cv2.line(cur_img_array, (240, 300), (240-(90-int(y)), 200), (0, 0, 255), 3)
            # Can determine the motor our with a simple exponential equation
            # x = abs(servo_out-90)
            # motor_out = (7.64*e^(-0.096*x)) - 1
            # motor_out = 90 - motor_out
            '''
            x_ = abs(servo_out - 90)
            motor_out = (7.64*np.e**(-0.096*x_)) - 1
            motor_out = int(80 - motor_out) # Only wanna go forwards
            cv2.line(cur_img_array, (100, 160), (100, 160-(90-motor_out)), raw_motor_to_rgb(motor_out), 3)
            print(motor_out, cur_motor)
            '''
        # Show frame
        # Convert to BGR cause thats how OpenCV likes it
        cv2.imshow('frame', cv2.cvtColor(cur_img_array, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    