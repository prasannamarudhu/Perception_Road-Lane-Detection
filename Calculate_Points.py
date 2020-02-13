import numpy as np
import cv2



def mouse_handler(event, x, y, flags, data):
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

                 
def get_four_points(im):
    
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
   
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    
    points = np.vstack(data['points']).astype(float)
    
    return points


mtx = np.matrix([[1154,0,671], [ 0,1148,386],[  0,0,1]])
dist =np.array([ -2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02])

print("Choose from the selected options for Videos")
print("press 1 for Project Video")
print("press 2 for Challenge Video")
print("")
I = int(input("Make your selection: "))
if I == 1:
    cap = cv2.VideoCapture('project_video.mp4')
elif I == 2:
    cap = cv2.VideoCapture('challenge_video.mp4')
else:
    print("Sorry selection could not be identified.")
    exit(0)

if (cap.isOpened()== False): 
   print("Error opening video stream or file")

count = 0
 
while(cap.isOpened()):
   
   ret, frame = cap.read()
   
   if ret ==True:
           count += 1
           if count == 348:
               cv2.imwrite("frame{:d}.jpg".format(count),frame)    
                             
               print("Select 4 points in a clockwise manner , starting from the top left corner. Press Enter after that.")
               im_src = cv2.imread("frame348.jpg")
               cv2.imshow("Image",im_src)
               
               pts_src = get_four_points(im_src)
               print(pts_src)


                         
   if ret ==False:
       break

cap.release()
cv2.destroyAllWindows()


            

       
 

