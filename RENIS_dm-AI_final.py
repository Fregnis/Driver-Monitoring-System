#**************************************************************************************
#
#   Driver Monitoring Systems using AI
#   Technologies for Autonomous Vehicles course @PoliTo
#   
#   First Assignment
#
#   File: dm-AI_final.m
#   Author: Francesco Renis
#   Company: Politecnico di Torino (student)
#   Date: 23 apr 2024
#
#**************************************************************************************

# 1 - Import the needed libraries
import cv2
import mediapipe as mp
import numpy as np 
import time
import statistics as st
import os

# 2 - Set the desired setting
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Enables  detailed eyes points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

## Get the list of available capture devices (comment out)
#index = 0
#arr = []
#while True:
#    dev = cv2.VideoCapture(index)
#    try:
#        arr.append(dev.getBackendName)
#    except:
#        break
#    dev.release()
#    index += 1
#print(arr)

# 3 - Open the video source
cap = cv2.VideoCapture(0) # Local webcam (index start from 0)
drowzy_timeout = 0
drowzy_msg_timeout = 0
distracted_timeout = 0
distracted_msg_timeout = 0
init = 0

# 4 - Iterate (within an infinite loop)
start = time.time()
while cap.isOpened(): 
    
    # 4.1 - Get the new frame
    success, image = cap.read()   

    # Also convert the color space from BGR to RGB
    if image is None:
        break
        #continue
    #else: #needed with some cameras/video input format
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performace
    image.flags.writeable = False
    
    # 4.2 - Run MediaPipe on the frame
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Convert the color space from RGB to BGR
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    point_RER = [] # Right Eye Right
    point_REB = [] # Right Eye Bottom
    point_RELB = [] # Right Eye Lid Bottom
    point_REL = [] # Right Eye Left
    point_RET = [] # Right Eye Top
    point_RELT = [] # Right Eye Lid Top

    point_LER = [] # Left Eye Right
    point_LEB = [] # Left Eye Bottom
    point_LELB = [] # Left Eye Lid Bottom
    point_LEL = [] # Left Eye Left
    point_LET = [] # Left Eye Top
    point_LELT = [] # Left Eye Lid Top

    point_REIC = [] # Right Eye Iris Center
    point_LEIC = [] # Left Eye Iris Center

    r_EAR_points = np.empty(6,dtype=object)
    l_EAR_points = np.empty(6,dtype=object)

    face_2d = []
    face_3d = []

    # 4.3 - Get the landmark coordinates

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):

                # Left eye indices list
                #LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
                #LEFT_IRIS = [473, 474, 475, 476, 477]
                # Right eye indices list
                #RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                #RIGHT_IRIS = [468, 469, 470, 471, 472]
                if idx == 23:
                    point_REB = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 27:
                    point_RET = (lm.x * img_w, lm.y * img_h)
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 33:
                    point_RER = (lm.x * img_w, lm.y * img_h)
                    r_EAR_points[0] = ((lm.x * img_w, lm.y * img_h))
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 133:
                    point_REL = (lm.x * img_w, lm.y * img_h)
                    r_EAR_points[3] = ((lm.x * img_w, lm.y * img_h))
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 144:
                    r_EAR_points[5] = ((lm.x * img_w, lm.y * img_h))
                if idx == 145:
                    point_RELB = ((lm.x * img_w, lm.y * img_h))
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)
                if idx == 153:
                    r_EAR_points[4] = ((lm.x * img_w, lm.y * img_h))
                if idx == 158:
                    r_EAR_points[2] = ((lm.x * img_w, lm.y * img_h))
                if idx == 159:
                    point_RELT = ((lm.x * img_w, lm.y * img_h))
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)
                if idx == 160:
                    r_EAR_points[1] = ((lm.x * img_w, lm.y * img_h))
                if idx == 253:
                    point_LEB = ((lm.x * img_w, lm.y * img_h))
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 257:
                    point_LET = ((lm.x * img_w, lm.y * img_h))
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 263:
                    point_LEL = (lm.x * img_w, lm.y * img_h)
                    l_EAR_points[3] = ((lm.x * img_w, lm.y * img_h))
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 362:
                    point_LER = (lm.x * img_w, lm.y * img_h)
                    l_EAR_points[0] = ((lm.x * img_w, lm.y * img_h))
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 0, 255), thickness=-1)
                if idx == 373:
                    l_EAR_points[4] = ((lm.x * img_w, lm.y * img_h))
                if idx == 374:
                    point_LELB = ((lm.x * img_w, lm.y * img_h))
                if idx == 380:
                    l_EAR_points[5] = ((lm.x * img_w, lm.y * img_h))
                if idx == 385:
                    l_EAR_points[1] = ((lm.x * img_w, lm.y * img_h))                    
                if idx == 387:
                    point_LELT = ((lm.x * img_w, lm.y * img_h))
                    l_EAR_points[2] = ((lm.x * img_w, lm.y * img_h))
                if idx == 468:
                    point_REIC = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 255, 0), thickness=-1)                    
                if idx == 469:
                    point_469 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)
                if idx == 470:
                    point_470 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)
                if idx == 471:
                    point_471 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)
                if idx == 472:
                    point_472 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 0), thickness=-1)
                if idx == 473:
                    point_LEIC = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(0, 255, 255), thickness=-1)
                if idx == 474:
                    point_474 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                if idx == 475:
                    point_475 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                if idx == 476:
                    point_476 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)
                if idx == 477:
                    point_477 = (lm.x * img_w, lm.y * img_h)
                    #cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)

                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                    

            # Convert into numpy arrays
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Calcuate camera matrix
            focal_length = 2 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            
            dist_matrix = np.zeros((4,1), dtype=np.float64)

            ## Calculate head gaze based on 3d points
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            # Get rotational matrices
            rmat, jac = cv2.Rodrigues(rot_vec)
        
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
            # Convert angles in degrees
            pitch = angles[0] * 1800
            yaw = -angles[1] * 1800
            roll = 180 + (np.arctan2(point_RER[1] - point_LEL[1], point_RER[0] - point_LEL[0]) * 180 / np.pi)
            if roll > 180:
                roll = roll - 360

            # Display head gaze angles
            cv2.putText(image, f"HEAD Roll: {roll:.2f} Pitch: {pitch:.2f} Yaw: {yaw:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
            
            # Display directions (nose) image not mirrored
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] - yaw * 2), int(nose_2d[1] - pitch * 2))
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            head_distracted = 0
            if abs(roll) >= 30 or abs(pitch) >= 30 or abs(yaw) >= 30:
                head_distracted = 1

            ## Calculate eyes gaze based on 2d points ##
            width_fraction = 0.5
            height_fraction = 0.4
            down_gaze_fraction = 0.3

            # compute some eye parameters
            r_eye_height = point_REB[1] - point_RET[1]
            l_eye_height = point_LEB[1] - point_LET[1]

            r_eye_center = [(point_REB[0] + point_RET[0])/2, (point_REL[1] + point_RER[1])/2]
            l_eye_center = [(point_LEB[0] + point_LET[0])/2, (point_LEL[1] + point_LER[1])/2]
            # calculate vertical distance between upper eye lid and eyebrow
            r_lid_eyebrow_dist = point_RELT[1] - point_RET[1]
            l_lid_eyebrow_dist = point_LELT[1] - point_LET[1]

            cv2.putText(image, f"R_lid_eyebrow_dist: {r_lid_eyebrow_dist:.2f} L_lid_eyebrow_dist: {l_lid_eyebrow_dist:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 

            # Draw eye and iris center
            cv2.circle(image, (int(point_LEIC[0]), int(point_LEIC[1])), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.circle(image, (int(point_REIC[0]), int(point_REIC[1])), radius=2, color=(0, 255, 0), thickness=-1)
            
            cv2.circle(image, (int(l_eye_center[0]), int(l_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1)
            cv2.circle(image, (int(r_eye_center[0]), int(r_eye_center[1])), radius=2, color=(128, 128, 128), thickness=-1) 
            
            # Define ellipses
            r_semiaxes = (int(abs(point_469[0]-point_471[0])*0.5 * width_fraction), int(abs(point_472[1]-point_470[1])*0.5 * height_fraction))   
            cv2.ellipse(image, (int(r_eye_center[0]), int(r_eye_center[1])), r_semiaxes, 0, 0, 360, color=(255, 255, 255), thickness=1)
            r_a = r_semiaxes[0]
            r_b = r_semiaxes[1] 
            r_xc = int(r_eye_center[0])
            r_yc = int(r_eye_center[1])
            r_x = point_REIC[0]
            r_y = point_REIC[1]
            
            l_semiaxes = (int(abs(point_474[0]-point_476[0])*0.5 * width_fraction), int(abs(point_477[1]-point_475[1])*0.5 * height_fraction))   
            cv2.ellipse(image, (int(l_eye_center[0]), int(l_eye_center[1])), l_semiaxes, 0, 0, 360, color=(255, 255, 255), thickness=1)
            l_a = l_semiaxes[0]
            l_b = l_semiaxes[1]
            l_xc = int(l_eye_center[0])
            l_yc = int(l_eye_center[1])
            l_x = point_LEIC[0]
            l_y = point_LEIC[1]

            ## Compare with thresholds
            cv2.putText(image, f"R_eye_height_limit: {0.4*r_eye_height:.2f} L_eye_height_limit: {0.4*l_eye_height:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 

            # RIGHT EYE
            r_pitch = 0
            r_yaw = 0
            if r_y >= r_yc + r_b*down_gaze_fraction or r_lid_eyebrow_dist > 0.4*r_eye_height:
                r_pitch = -1
            if r_y <= r_yc - r_b:
                r_pitch = 1
            if r_x >= r_xc + r_a:
                r_yaw = 1
            if r_x <= r_xc - r_a:
                r_yaw = -1
            cv2.putText(image, f"R_EYE Pitch: {r_pitch:.2f} Yaw: {r_yaw:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 


            # LEFT EYE
            l_pitch = 0
            l_yaw = 0
            if l_y >= l_yc+ l_b*down_gaze_fraction or l_lid_eyebrow_dist > 0.4*l_eye_height:
                l_pitch = -1
            if l_y <= l_yc - l_b:
                l_pitch = 1
            if l_x >= l_xc + l_a:
                l_yaw = 1
            if l_x <= l_xc - l_a:
                l_yaw = -1
            cv2.putText(image, f"L_EYE Pitch: {l_pitch:.2f} Yaw: {l_yaw:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
            
            eyes_distracted = 0
            if abs(r_pitch) >= 1 or abs(l_pitch) >= 1 or abs(r_yaw) >= 1 or abs(l_yaw) >= 1:
                eyes_distracted = 1


            ## Calculate EAR ## 
            r_EAR = (abs(r_EAR_points[1][1] - r_EAR_points[5][1]) + abs(r_EAR_points[2][1] - r_EAR_points[4][1])) / (2*abs(r_EAR_points[0][0] - r_EAR_points[3][0]))
            l_EAR = (abs(l_EAR_points[1][1] - l_EAR_points[5][1]) + abs(l_EAR_points[2][1] - l_EAR_points[4][1])) / (2*abs(l_EAR_points[0][0] - l_EAR_points[3][0]))
            avg_EAR = 0.5*(r_EAR + l_EAR)
            open_EAR = 0.28 # statically set

            # check if eyes are kept open or kept close
            eyes_timeout = 0
            if avg_EAR >= 0.8*open_EAR or avg_EAR <= 0.7*open_EAR:
                eyes_timeout = 1

            #cv2.putText(image, f"Right EAR: {r_EAR:.2f} Left EAR: {l_EAR:.2f}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
            cv2.putText(image, f"Avg EAR: {avg_EAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 


            # speed reduction (comment out for full speed)
            #time.sleep(1/200) # [s]

        end = time.time()
        totalTime = end-start
        start = time.time()

        if totalTime>0:
            fps = 1 / totalTime
        else:
            fps=0
        
        # Detect if driver is distracted and eventually print alarm message
        if head_distracted or eyes_distracted: 
            distracted_timeout += totalTime
        else:
            distracted_timeout = 0
        cv2.putText(image, f'Distracted time: {int(distracted_timeout)}', (10,440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if distracted_timeout >= 3: # if driver is distracted for more than 3 seconds
          distracted_msg_timeout = 2 * fps # Message stays on for 2 seconds
        
        if distracted_msg_timeout > 0:
            distracted_msg_timeout-=1
            cv2.putText(image, f'WARNING: Driver is distracted', (10,250), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 4)


        # Calculate simplified PERCLOSE and aventually print alarm message
        if eyes_timeout: 
            drowzy_timeout += totalTime
        else:
            drowzy_timeout = 0
        cv2.putText(image, f'Drowzy time: {int(drowzy_timeout)}', (10,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if drowzy_timeout >= 10: # if eyes are kept open or kept close for more than 10 seconds
          drowzy_msg_timeout = 5 * fps # Message stays on for 5 seconds
        
        if drowzy_msg_timeout > 0:
            drowzy_msg_timeout-=1
            cv2.putText(image, f'WARNING: Driver is drowzy', (30,300), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 4)

        #print("FPS:", fps)
        cv2.putText(image, f'FPS : {int(fps)}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # 4.5 - Show the frame to the user
        cv2.imshow('Technologies for Autonomous Vehicles - Driver Monitoring Systems using AI Assignement', image)       
                    
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 5 - Close properly soruce and eventual log file
cap.release()
#log_file.close()

# [EOF]
