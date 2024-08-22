import cv2
def detect_bounding_box(vid):
    # Detect and Blur using Haarcascade
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml" 
    )
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

        roi = vid[y:y+h, x:x+w] 

        # applying a gaussian blur 
        roi = cv2.GaussianBlur(roi, (0, 0), 30) 

        vid[y:y+roi.shape[0], x:x+roi.shape[1]] = roi 
    return faces

if(__name__=="__main__"):

    video_capture = cv2.VideoCapture(0)

    while True:

        result, video_frame = video_capture.read()  
        if result is False:
            break 

        faces = detect_bounding_box(
            video_frame
        )  

        cv2.imshow(
            "Real Time Face Blur", video_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()