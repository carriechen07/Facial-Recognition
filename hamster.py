import cv2
import mediapipe as mp
import time
import math

#setup
model_path = 'face_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#options
img_options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.IMAGE)
vid_options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path), running_mode=VisionRunningMode.VIDEO)

#video source
cap = cv2.VideoCapture(0)

#main container
face_database = {}

with FaceLandmarker.create_from_options(img_options) as img_landmarker, \
     FaceLandmarker.create_from_options(vid_options) as vid_landmarker:

    #helper
    def get_landmarks(img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = img_landmarker.detect(mp_img)
        if res.face_landmarks:
            return [(pt.x, pt.y) for pt in res.face_landmarks[0]]
        return None

    #default image
    states = {
        "Default": ["default.jpg", "hamster_default.jpg"],
        "Jolly": ["jolly.jpg", "hamster_jolly.jpg"],
        "Disapprove": ["disapprove.jpg", "hamster_disapprove.jpg"],
        "Eyebrow": ["kiss.jpg", "hamster_kiss.jpg"], 
        "Smiley": ["smiley.jpg", "hamster_smiley.jpg"]
    }

    for name, files in states.items():
        data = get_landmarks(files[0])
        display_img = cv2.imread(files[1])
        if data:
            face_database[name] = [data, display_img]

    #get camera
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        h, w, _ = frame.shape
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        vid_result = vid_landmarker.detect_for_video(mp_frame, int(time.time() * 1000))

        #draw video and compare
        if vid_result.face_landmarks:
            for landmarks in vid_result.face_landmarks:
                live_list = []
                for i in range(468):
                    pt = landmarks[i]
                    live_list.append((pt.x, pt.y))
                
                live_cx, live_cy = live_list[1]
                best_match = None
                lowest_diff = float('inf')

                #calculating what is closest
                for name, content in face_database.items():
                    ref_data = content[0]
                    ref_cx, ref_cy = ref_data[1]
                    
                    total_diff = 0
                    for i in range(468):
                        rdx, rdy = ref_data[i][0] - ref_cx, ref_data[i][1] - ref_cy
                        rlx, rly = live_list[i][0] - live_cx, live_list[i][1] - live_cy
                        total_diff += math.sqrt((rdx - rlx)**2 + (rdy - rly)**2)

                    if total_diff < lowest_diff:
                        lowest_diff = total_diff
                        best_match = name

                #show
                if best_match:
                    cv2.imshow("Hamster", face_database[best_match][1])
                
                #for pt in live_list:
                    #cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 1, (50, 100, 0), -1)

        cv2.imshow("Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()