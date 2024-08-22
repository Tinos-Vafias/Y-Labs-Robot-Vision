import pyrealsense2 as realsense
import numpy as np
import cv2
from ultralytics import YOLO
import speech_recognition as speech
import threading
from pocketsphinx import LiveSpeech, get_model_path
import os
# Set paths for models and dictionary
model_path = get_model_path()
acoustic_model_path = os.path.join(model_path, 'en-us\en-us')
language_model_path = os.path.join(model_path, 'en-us\en-us.lm.bin')
dictionary_path = os.path.join(model_path, 'en-us\cmudict-en-us.dict')


# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Initialize RealSense pipeline
pipe = realsense.pipeline()
confg = realsense.config()
confg.enable_stream(realsense.stream.color, 640, 480, realsense.format.bgr8, 30)
confg.enable_stream(realsense.stream.depth, 640, 480, realsense.format.z16, 30)
pipe.start(confg)

# Global variables
locked = False
locked_object = None
recognizer = speech.Recognizer()
microphone = speech.Microphone()
desired = ''
locked_id = None

#kalman filter parameters
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.07
lost_frame_count = 0  # Counter to keep track of lost frames
max_lost_frames = 10  # Maximum number of frames to keep predicting before stopping

#method for spech recognition through CMU PocketSphinx
def listen_for_command():
    global desired, locked, locked_object
    try:
        with microphone as source:
            while True:
                print("Listening for command...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                command = recognizer.recognize_sphinx(audio).lower()
                print(f"Recognized command: {command}")
                if "lock" in command:
                    locked = True
                    print("Locking onto current object...")
                elif "stop" in command:
                    locked = False
                    locked_object = None
                    print("Unlocked.")
                else:
                    # Update desired class based on command
                    for label in model.names.values():
                        if label in command:
                            desired = label
                            locked = False
                            lost_frame_count = 0
                            locked_object = None
                            locked_id = None
                            print(f"Desired class set to: {desired}")

    except Exception as e:
        print(f"Error recognizing command: {e}")
        
def get_distance_and_coordinates(bbox, depth_frame):
    x, y, w, h, label, conf, id = bbox
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    if 0 <= center_x < depth_frame.get_width() and 0 <= center_y < depth_frame.get_height():
        distance = depth_frame.get_distance(center_x, center_y)
        # Get intrinsic coords of the depth stream
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        # Calculate real-world coordinates
        depth_point = realsense.rs2_deproject_pixel_to_point(intr, [center_x, center_y], distance)
        return distance, depth_point
    else:
        return None, None
    
def detect_objects(frame, depth):
    global locked, locked_object, desired, lost_frame_count, max_lost_frames, locked_id
    results = model.track(frame, persist= True)
    current_objects = []
    
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = detection.xyxy[0]
            conf = detection.conf[0]
            cls = detection.cls[0]
            label = model.names[int(cls)]
            id = int(detection.id[0]) if detection.id is not None else None
            print(id)
            if label ==  desired:
                current_objects.append((x1, y1, x2, y2, label, conf, id))
                if not locked and label == desired:
                    label_position_y = int(y1) - 10 if y1 - 10 > 10 else int(y1) + 10
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f'{desired} {conf:.2f}', (int(x1), label_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                #print(f"Detected desired object: {label} at {x1}, {y1}, {x2}, {y2}")

    if locked:
        if locked_object is not None:
            # Find the object closest to the previous locked position
            min_dist = float('inf')
            prediction = kalman.predict()
            pred_x1, pred_y1 = prediction[0], prediction[1]
            if current_objects:
                for obj in current_objects:
                    if obj[6] == locked_id:
                        # Update the Kalman filter with the detected object's position
                        new_locked_object = min(current_objects, key=lambda obj: np.linalg.norm(np.array([(obj[0] + obj[2]) / 2, (obj[1] + obj[3]) / 2]) - np.array([pred_x1, pred_y1])))
                        locked_object = new_locked_object
                        lost_frame_count = 0
                        measurement = np.array([[np.float32((locked_object[0] + locked_object[2]) / 2)], [np.float32((locked_object[1] + locked_object[3]) / 2)]])
                        kalman.correct(measurement)
                        distance, coords = get_distance_and_coordinates(locked_object, depth)
                        cv2.rectangle(frame, (int(locked_object[0]), int(locked_object[1])), (int(locked_object[2]), int(locked_object[3])), (0, 255, 0), 2)
                        if distance is not None and coords is not None:
                            label_position_y = int(locked_object[1]) - 10 if locked_object[1] - 10 > 10 else int(locked_object[1]) + 10
                            cv2.putText(frame, f'{desired} {conf:.2f} Dist: {distance:.2f}m ID:{locked_id}', (int(locked_object[0]), int(locked_object[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            print(f"Distance: {distance} meters, Coordinates: {coords}")
                            print(f"Locked object updated: {locked_object}")
                        break
                # Draw the updated locked object box
                
                
            else:
                lost_frame_count += 1
                if lost_frame_count > max_lost_frames:
                    locked = False
                    locked_object = None
                    lost_frame_count = 0
                    locked_id = None
                    print("Object lost, stopping tracking.")
                else:
                    # Draw the Kalman filter prediction box
                    cv2.rectangle(frame, (int(pred_x1 - (locked_object[2] - locked_object[0]) / 2), int(pred_y1 - (locked_object[3] - locked_object[1]) / 2)),
                    (int(pred_x1 + (locked_object[2] - locked_object[0]) / 2), int(pred_y1 + (locked_object[3] - locked_object[1]) / 2)), (0, 255, 0), 2) 
                    
        else:
            # Initialize the locked object
            for (x1, y1, x2, y2, label, conf, obj_id) in current_objects:
                if label == desired:
                    locked_id = obj_id
                    locked_object = (x1, y1, x2, y2, label, conf, obj_id)
                    kalman.statePre = np.array([x1, y1, 0, 0], np.float32)
                    kalman.statePost = np.array([x1, y1, 0, 0], np.float32)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    print(f"Locked onto new object: {locked_object}")
                    break

def main():
    #speech recognition run in parallel
    threading.Thread(target=listen_for_command, daemon=True).start()
    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            dframe = np.asanyarray(depth_frame.get_data())
            detect_objects(frame, depth_frame)

            cv2.imshow('Object Detection', frame)
            #hit q to quit program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
