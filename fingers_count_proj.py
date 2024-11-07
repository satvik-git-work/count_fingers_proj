import cv2
import numpy as np

from sklearn.metrics import pairwise #used for distance calculations

bg=None
accumulated_weight=0.5 #should be between 0 and 1, so a middle value of 0.5

roi_top=60
roi_bottom=300
roi_left=300
roi_right=600

#function that finds the average bg value

def calc_accum_avg(frame,accumulated_weight):
    global bg
    if bg is None:
        bg=frame.copy().astype('float')
        return None
    cv2.accumulateWeighted(frame,bg,accumulated_weight)


#thresholding to grab hand segment from the ROI

def segment(frame,threshold=25):
    diff=cv2.absdiff(bg.astype('uint8'),frame)
    ret,thresholded=cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY)
    contours,hierarchy=cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        return None
    else: #assuming largest external contour within the ROI should be only the hand
        hand_segment=max(contours,key=cv2.contourArea)
        return (thresholded,hand_segment)

#drawing a convex hull for figuring out the fingers

def count_fingers(thresholded,hand_segment):
    conv_hull=cv2.convexHull(hand_segment)
    top=tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
    bottom=tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
    left=tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
    right=tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

    cx=(left[0]+right[0])//2
    cy=(top[1]+bottom[1])//2

    distance=pairwise.euclidean_distances([(cx,cy)],Y=[left,right,top,bottom])[0]
    max_distance=distance.max()
    rad=int(0.90*max_distance)
    circum=2*np.pi*rad

    circular_roi=np.zeros(thresholded.shape[:2],dtype='uint8')
    cv2.circle(circular_roi,(cx,cy),rad,255,100)

    circular_roi=cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)

    contours,hierarchy=cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    count=0

    for c in contours:
        (x,y,w,h)=cv2.boundingRect(c)

        out_of_wrist=(cy+(cy*0.25))>(y+h)
        limit_pts=((circum*0.25)>c.shape[0])
        if out_of_wrist and limit_pts:
            count+=1
    
    return count

cam=cv2.VideoCapture(0)
num_frames=0
while True:
    ret,frame=cam.read()
    frame_copy=frame.copy()
    roi=frame[roi_top:roi_bottom,roi_left:roi_right]

    gs=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gs=cv2.GaussianBlur(gs,(7,7),0)

    if num_frames<60:
        calc_accum_avg(gs,accumulated_weight)
        if num_frames<=59:
            cv2.putText(frame_copy,"Getting backgound, waiti!",(200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(10,20,30),2)
            cv2.imshow('finger count',frame_copy)
    else:
        hand=segment(gs)

        if hand is not None:
            thresholded,hand_segment=hand

            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(200,220,240),3)

            fingers=count_fingers(thresholded,hand_segment)
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(100,150,200),2)
            cv2.imshow("thresholded",thresholded)


    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(24,48,84),2)
    num_frames+=1
    cv2.imshow("finger count",frame_copy)

    if cv2.waitKey(1) & 0xFF ==27 :
        break

cam.release()
cv2.destroyAllWindows()




    




