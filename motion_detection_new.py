# import the necessary packages
from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2

# specify the video file path and minimum area size directly
video_path = "./videos/motion_test.mp4" 
min_area = 500
output_path = "./videos/motion_output.mp4"

# open the video file
vs = cv2.VideoCapture(video_path)

# initialize the first frame in the video stream
firstFrame = None

# get the video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = None

# loop over the frames of the video
while True:
    # grab the current frame
    ret, frame = vs.read()
    text = "Unoccupied"

    # if the frame could not be grabbed, then we have reached the end of the video
    if not ret:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        # initialize the video writer after the first frame is grabbed
        (h, w) = frame.shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
        continue

    # compute the absolute difference between the current frame and first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame, and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    # draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # write the frame to the output video file
    out.write(frame)

    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.release()
out.release()
cv2.destroyAllWindows()
