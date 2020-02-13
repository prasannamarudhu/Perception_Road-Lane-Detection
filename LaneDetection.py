"""
*****************  Team Members: PRASANNA MARUDHU BALASUBRAMANIAN (116197700) , SAKET SESHADRI GUDIMETLA (116332293), CHAYAN KUMAR PATODI (116327428)  **********************

*****************  Class: Robotics - 2019 Spring Semester             ********************
*****************  Course: ENPM673-Perception for Autonomous Robots   ********************

######################################################################################################################################
 
* This Python program is developed to Detect the Lane on the Road and predict the turn direction to indicate the type of turn ahead  *
* This program is made common for both the Project_video and the challenge_video *

######################################################################################################################################

"""


# Modules used in the code are listed below #

import numpy as np
import cv2

# The calibration matrix and the distortion matrix given to us are used below #

mtx = np.matrix([[1.15422732e+03,0,6.71627794e+02], [  0,1.14818221e+03,3.86046312e+02],[  0,0,1]])
dist =np.array([ -2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02])

# Assigning average prediction length #
Avg_Predic_Length = 10

###Capturing the frames of the given video using the VideoCapture function by accepting the input from the user

print("Select anyone of the video below : ")
print("press 1 for Project_video")
print("press 2 for challenge_video")
print("")

#Accepting input
option = int(input("Make your selection: "))
if option == 1:
    cap = cv2.VideoCapture('project_video.mp4')
elif option == 2:
    cap = cv2.VideoCapture('challenge_video.mp4')
else:
    #Printing entry of wrong input to indicate user
    print("Please select either 1 or 2....Stopping the execution now")
    exit(0)

###Checking whether it is possible to open the video and if not then the error message is displayed
if (cap.isOpened()== False): 
    print("Error opening video stream or file") 	#Error message printing


def main():
   
    ### Reading until video is completed
    count =0
    while(cap.isOpened()==True):
    
        ret , frame1 = cap.read()
        
	#Undistorting the given video frame
        frame1 = cv2.undistort(frame1, mtx, dist,None,mtx)
        frame2 = frame1
        
        count = count+1

        if count >= 110 and count <= 160:
            frame1 = adjust_gamma(frame1, gamma=-1.0)
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
            
            
        else:
            frame1=frame1

   	# Converting image to binary
        img_b = image_binary(frame1)

        # Choosing the Region of Interest
        lower_left = [0,720] 
        lower_right = [1280,720]
        top_left = [0,360] 
        top_right = [1280,360] 
        vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
        frame = region_of_interest(frame1, vertices)
        #cv2.imshow('roi',frame)


 # ---------------------------- Perspective Transform --------------------------#
	
        # Feeding the source points calculated from the Calculate_Points.py file here 

        line_dst_offset = 200
        src = [632, 470], \
              [698, 470], \
              [990, 696], \
              [302, 696]
        
        dst = [src[3][0] + line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, 0], \
              [src[2][0] - line_dst_offset, src[2][1]], \
              [src[3][0] + line_dst_offset, src[3][1]]

	# The binary image is warped using the source and destination point
        img_w = warp(img_b, src, dst)
       
        # calcualting the left and right fit points
        try:
            left_fit, right_fit = fit_from_lines(left_fit, right_fit, img_w)

            mov_avg_left = np.append(mov_avg_left,np.array([left_fit]), axis=0)
            mov_avg_right = np.append(mov_avg_right,np.array([right_fit]), axis=0)

        except:
	    #Calling the sliding windows function
            left_fit, right_fit = sliding_window(img_w)

            mov_avg_left = np.array([left_fit])
            mov_avg_right = np.array([right_fit])

        left_fit = np.array([np.mean(mov_avg_left[::-1][:,0][0:Avg_Predic_Length]),
                             np.mean(mov_avg_left[::-1][:,1][0:Avg_Predic_Length]),
                             np.mean(mov_avg_left[::-1][:,2][0:Avg_Predic_Length])])
        right_fit = np.array([np.mean(mov_avg_right[::-1][:,0][0:Avg_Predic_Length]),
                             np.mean(mov_avg_right[::-1][:,1][0:Avg_Predic_Length]),
                             np.mean(mov_avg_right[::-1][:,2][0:Avg_Predic_Length])])

        if mov_avg_left.shape[0] > 1000:
            mov_avg_left = mov_avg_left[0:Avg_Predic_Length]
        if mov_avg_right.shape[0] > 1000:
            mov_avg_right = mov_avg_right[0:Avg_Predic_Length]

        #Displaying the final output image 
	final = draw_lines(frame2, img_w, left_fit, right_fit, perspective=[src,dst])
        
        cv2.imshow('final', final)

        #Quiting output window when the letter 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Releasing the capturing of frames from the video 
    cap.release()
    cv2.destroyAllWindows()        


# The area of interest is chosen using this function
def region_of_interest(img, vertices):

    """
    An image mask is applied in the original frame.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    """

    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def image_binary(img, sobel_kernel=7, mag_thresh=(3, 255), s_thresh=(170, 255)):

    # *********** Binary Thresholding ************ #
    # Binary Thresholding is an intermediate step to improve lane line perception and it includes image transformation to gray scale to apply sobel transform and binary slicing to output 0,1 type images 	 according to pre-defined threshold. 
    # Also it's performed RGB to HSV transformation to get S information which intensifies lane line detection.
    # The output is a binary image combined with best of both S transform and magnitude thresholding.

    #Converting image to gray image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Image conversion to hsv
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    s_channel = gray
    
    #Yellow lane mark detection 
    lower_yellow = np.array([2,20,20])
    upper_yellow = np.array([30,255,255])
    mask_YELLOW = cv2.inRange(hsv,lower_yellow,upper_yellow)
       
    res_YELLOW = cv2.bitwise_and(img,img, mask = mask_YELLOW) 
    FiltMedian_YELLOW=cv2.medianBlur(res_YELLOW,3)  # Filter, Smooth, Blur
    #Count and print
    gray_YELLOW = cv2.cvtColor(FiltMedian_YELLOW,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gy',gray_YELLOW)
        
    im_bw1 = cv2.threshold(gray_YELLOW,100,255,cv2.THRESH_BINARY)[1]
    #cv2.imshow('bw',im_bw1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(80,80))
    res_YELLOW = cv2.morphologyEx(im_bw1,cv2.MORPH_OPEN,kernel)
    ret , threshold_YELLOW = cv2.threshold(res_YELLOW,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, contoursYELLOW,_=cv2.findContours (threshold_YELLOW,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    YelContour = cv2.drawContours(FiltMedian_YELLOW,contoursYELLOW,-1,(50,255,100),2)
    ColMerge = cv2.addWeighted(img, 1, YelContour, 1, 0)    
    

    # Binary matrixes creation
    sobel_binary = np.zeros(shape=gray.shape, dtype=bool)
    s_binary = sobel_binary
    combined_binary = s_binary.astype(np.float32)

    # Sobel Transform
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel_abs = np.abs(sobelx**2 + sobely**2)
    sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))

    sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1

    # Threshold color channel
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combining the two binary thresholds

    combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))


    # ---------------- MASKED IMAGE -------------------- #
    offset = 100
    mask_polyg = np.array([[(0 + offset, img.shape[0]),
                            (img.shape[1] / 2.5, img.shape[0] / 1.65),
                            (img.shape[1] / 1.8, img.shape[0] / 1.65),
                            (img.shape[1], img.shape[0])]],
                          dtype=np.int)


    mask_img = np.zeros_like(combined_binary)
    
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    # Applying polygon
    cv2.fillPoly(mask_img, mask_polyg, ignore_mask_color)
    masked_edges = cv2.bitwise_and(combined_binary, mask_img)
    img_b = cv2.addWeighted(masked_edges, 1, im_bw1, 1, 0)
    #cv2.imshow('img_b ', img_b)
    
    return img_b

# Warping the input image with source and destination points
def warp(img, src, dst):

    src = np.float32([src])
    dst = np.float32([dst])
    
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                               dsize=img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)


#Sliding window calculation to guide drawing the line on the lane markings 
def sliding_window(img_w):
    
    #histogram calculation
    histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
   
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img_w, img_w, img_w)) * 255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    #midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:640])
    rightx_base = np.argmax(histogram[640:]) + 640

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windowsimg_wimg_w
    window_height = np.int(img_w.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_w.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_w.shape[0] - (window + 1) * window_height
        win_y_high = img_w.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (50, 255, 100), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (50, 255, 100), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit


def fit_from_lines(left_fit, right_fit, img_w):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img_w.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

#The lines are drwan using this function on the lane markings
def draw_lines(img, img_w, left_fit, right_fit, perspective):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img_w).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    #color_warp_center = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    pts_center = ((pts_right-pts_left)/2)
    print ('pts_left', pts_left[0][0][0]) 

    #Detecting the direction of the road ahead
    if pts_left[0][0][0] <= 208:
        Direction = "Left Turn Ahead"
    elif pts_left[0][0][0] > 208 and pts_left[0][0][0] <= 280:
        Direction = "Straight Road Ahead"
    elif pts_left[0][0][0] >= 390 and pts_left[0][0][0] < 480:
        Direction = "Straight Road Ahead"
    elif pts_left[0][0][0] > 280 and pts_left[0][0][0] <= 350:
        Direction = "Right Turn Ahead"
    elif pts_left[0][0][0] >= 480:
        Direction = "Right Turn Ahead"
    else:
        Direction=" "

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp_center, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts]), (55, 250, 50))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp(color_warp, perspective[1], perspective[0])
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)
    #color warping the lines
    color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
    cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(10,255,175), thickness=20)
    cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(10,255,175), thickness=20)
    newwarp_lines = warp(color_warp_lines, perspective[1], perspective[0])
    #combining the newwarped line and the result
    result = cv2.addWeighted(result, 1, newwarp_lines, 1, 0)

    # Road curvature calculation #

    img_height = img.shape[0]
    y_eval = img_height

    ym_per_pix = 30 / 720.  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty = np.linspace(0, img_height - 1, img_height)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    radius = round((float(left_curverad) + float(right_curverad))/2.,2)

    # --- Printing text on screen ------ #
    text = " Direction = %s \n\n Radius = %s [m] " % (str(Direction), str(radius))
    

    for i, line in enumerate(text.split('\n')):
        i = 650 + 20 * i
        cv2.putText(result, line, (550,i), cv2.FONT_HERSHEY_DUPLEX, 0.75,(255,10,100),1,cv2.LINE_AA)
    return result

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

if __name__ == '__main__':
    main()
