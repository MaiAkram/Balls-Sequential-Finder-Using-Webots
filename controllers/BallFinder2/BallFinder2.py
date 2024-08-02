from controller import Robot, Camera
import cv2
import numpy as np

# Set up the robot
robot = Robot()
time_step = int(robot.getBasicTimeStep())

# Enable the camera
camera = robot.getDevice("camera")
camera.enable(time_step)

# Enable the motors
left_motor = robot.getDevice("motor_1")
right_motor = robot.getDevice("motor_2")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Define the arena corners
corners = [(1.5, 3.0), (1.5, 3.0), (-1.5, 3.0), (-1.5, 0.0)]

# Enable Distance Sensors
sensors = []
for i in range(3):
    sensor = robot.getDevice(f'ds{i}')
    sensors.append(sensor)
    sensors[i].enable(time_step)

# Define the colors of the balls
ball_colors = ["red", "blue", "green", "yellow"]

# Initialize variables
current_ball_index = 0
balls_found = []

# Set robot's speed
max_speed = 6.28

n = 0;
m = 0;
R = 1;

def color_detection(image):
    # Define color thresholds in RGB format
    red_lower = np.array([0, 0, 100], dtype=np.uint8)
    red_upper = np.array([50, 50, 255], dtype=np.uint8)
    green_lower = np.array([0, 100, 0], dtype=np.uint8)
    green_upper = np.array([50, 255, 50], dtype=np.uint8)
    blue_lower = np.array([100, 0, 0], dtype=np.uint8)
    blue_upper = np.array([255, 50, 50], dtype=np.uint8)
    yellow_lower = np.array([0, 100, 100], dtype=np.uint8)
    yellow_upper = np.array([50, 255, 255], dtype=np.uint8)

    # Threshold the image to get only desired colors
    red_mask = cv2.inRange(image, red_lower, red_upper)
    green_mask = cv2.inRange(image, green_lower, green_upper)
    blue_mask = cv2.inRange(image, blue_lower, blue_upper)
    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)

    # Bitwise-AND mask and original image
    red_result = cv2.bitwise_and(image, image, mask=red_mask)
    green_result = cv2.bitwise_and(image, image, mask=green_mask)
    blue_result = cv2.bitwise_and(image, image, mask=blue_mask)
    yellow_result = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Normalize the color arrays to range from 0 to 1
    red_array = red_result.astype(np.float32) / 255.0
    green_array = green_result.astype(np.float32) / 255.0
    blue_array = blue_result.astype(np.float32) / 255.0
    yellow_array = yellow_result.astype(np.float32) / 255.0

    # Check which color has the most pixels
    red_count = cv2.countNonZero(red_mask)
    green_count = cv2.countNonZero(green_mask)
    blue_count = cv2.countNonZero(blue_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    
    # Determine the dominant color
    if red_count > green_count and red_count > blue_count:
        return [1, 0, 0]  # Red
    elif green_count > red_count and green_count > blue_count:
        return [0, 1, 0]  # Green
    elif blue_count > red_count and blue_count > green_count:
        return [0, 0, 1]  # Blue
    elif yellow_count >= green_count and yellow_count >= red_count and yellow_count > blue_count:
        return [1, 1, 0]
    else:
        return "No dominant color detected"

# Main loop
while robot.step(time_step) != -1:
    # Get camera image
    imageI = camera.getImage()
    image = camera.getImageArray()
    col = "None"
    
    if image:
         # Convert image to numpy array
        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Perform color detection
        detected_color = color_detection(image)
            
        if (detected_color == [1, 0, 0]) and (ball_colors[current_ball_index] == "red"):# and (front_wall):
            balls_found.append("red")
            current_ball_index += 1
            col = "Red"
            print("First Ball (Red) Found!")
        elif (detected_color == [0, 0, 1]) and (ball_colors[current_ball_index] == "blue"):# and (front_wall):
            balls_found.append("blue")
            current_ball_index += 1
            col = "Blue"
            print("Second Ball (Blue) Found!")
        elif (detected_color == [0, 1, 0]) and (ball_colors[current_ball_index] == "green"):# and (front_wall):
            balls_found.append("green")
            current_ball_index += 1
            col = "Green"
            print("Third Ball (Green) Found!")
        elif (detected_color == [1, 1, 0]) and (ball_colors[current_ball_index] == "yellow"):
            balls_found.append("yellow")
            col = "yellow"
            print("Fourth Ball (yellow) Found!")
           
        #Sprint("Detected color:", col)
    # Display the image
        cv2.imshow("Camera Image", image)
        cv2.waitKey(1)  # Keep the window open until a key is pressed            

    # Check if all balls have been found
    if len(balls_found) == len(ball_colors):
        print("All four balls found!")
        break
    
    #  Check If Came to an End
    front_wall = sensors[2].getValue() > 200
     
    if (front_wall) or ((n>0) and (n<32)):
        front_wall = True
        n = n+1
        #print(n)
        #print(balls_found)
    elif n >= 32:
        #if(R==0):
        #    R = 1
        #else:
        #    R = 0
        n = 0
        m = 65
    if (m > 0):
        m = m-1
        if m == 0:
            if(R==0):
                R = 1
            else:
                R = 0
            
    # Turn Right in Place
    if ((front_wall) or ((m <= 32) and (m > 0))):# and (R == 1):
        print("Turn Right in Place")
        speed0 = max_speed
        speed1 = -max_speed
        
    # Turn Left in Place
    elif ((front_wall) or ((m <= 33) and (m > 0))) and (R == 0):
        print("Turn Left in Place")
        speed0 = -max_speed
        speed1 = max_speed
    # Move Forward
    else:
        speed0 = max_speed
        speed1 = max_speed
        print("Move Forward")
       
    # Move the robot forward
    left_motor.setVelocity(speed0)
    right_motor.setVelocity(speed1)
    
# Stop the robot
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)