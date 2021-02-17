import numpy as np
import cv2 as cv
from scipy.interpolate import UnivariateSpline # for problem 3
import math # for problem 4

def problem1(img_url, dark_coeff, blend_coeff, mode):
    img = cv.imread(img_url)
    rows,columns,channels = img.shape

    # parameter validation

    if dark_coeff < 0:
        dark_coeff = 0
    elif dark_coeff > 1:
        dark_coeff = 1

    if blend_coeff < 0:
        blend_coeff = 0
    elif blend_coeff > 1:
        blend_coeff = 1

    # darken the image

    img = img*(1-dark_coeff)

    # output this intermediate stage

    cv.imwrite("problem1_stage1.jpg", img)

    window_name = 'Problem 1 Stage 1'
    output_img = img.astype(np.uint8)
    cv.imshow(window_name, output_img)

    # decide on mode

    if mode == "light":
        mask = cv.imread("light_leak_mask.png")
    elif mode == "rainbow":
        mask = cv.imread("rainbow_leak_mask.png")
    else:
        print("Error: invalid mode - assuming light leak mode")
        mask = cv.imread("light_leak_mask.png")

    # resize the mask to fit the image

    dim = (columns, rows)
    mask = cv.resize(mask, dim, interpolation = cv.INTER_AREA)

    # add the mask

    for column in range(columns): # scans across the x-axis
        for row in range(rows): # scans down the y-axis
            for channel in range(channels):
                # add the image and mask pixels, weighted by blend_coeff
                new_pixel = int((1-blend_coeff) * img[row, column, channel] + blend_coeff * mask[row, column, channel])
                if new_pixel < 255: # to avoid overflow
                    img[row, column, channel] = new_pixel
                else:
                    img[row, column, channel] = 255


    # output the final result

    cv.imwrite("problem1_result.jpg", img)

    window_name = 'Problem 1 Result'
    output_img = img.astype(np.uint8)
    cv.imshow(window_name, output_img)

    # close all windows when the user presses any key
    cv.waitKey(0)
    cv.destroyAllWindows()



def problem2(img_url, blend_coeff, mode, noise, brightness):

    if mode == "colour":
        curr_channel = range(0,1) # apply the first mask to the first channel on the first iteration
        repeats = 2
    else: # assume mode == "greyscale"
        curr_channel = range(3) # apply the same mask to all three colour channels
        repeats = 1

    img = cv.imread(img_url)
    rows,columns,channels = img.shape

    for repeat in range(repeats):
        if repeat == 1: # then mode == colour and we're applying the second (and final) colour channel
            curr_channel = range(1,2) # apply the second mask to the second channel on the second iteration

            # also output the image with just one colour channel modified
            cv.imwrite("problem2_stage3.jpg", img)

            window_name = 'Problem 2 Stage 3'
            output_img = img.astype(np.uint8)
            cv.imshow(window_name, output_img)


        # generate the noise texture
        texture = np.random.normal(0, noise, (rows,columns,1)) # only one channel is needed for greyscale

        # output this intermediate stage

        cv.imwrite("problem2_stage1_"+str(repeat+1)+".jpg", texture)

        window_name = 'Problem 2 Stage 1 Iteration ' + str(repeat + 1)
        output_img = texture.astype(np.uint8)
        cv.imshow(window_name, output_img)

        # apply a motion blur effect to the noise texture
        mask_size = 19

        # generate the mask
        mask = np.zeros((mask_size, mask_size))
        mask[9, :7] = np.ones(7)
        mask[10, :13] = np.ones(13)
        mask[11, 7:] = np.ones(12)
        mask[12, 13:] = np.ones(6)
        

        mask = mask / mask_size

        # applying the kernel to the input image
        texture = cv.filter2D(texture, -1, mask, cv.BORDER_DEFAULT) # -1 defines the ddepth as being equal to the source

        # output this intermediate stage

        cv.imwrite("problem2_stage2_"+str(repeat+1)+".jpg", texture)

        window_name = 'Problem 2 Stage 2 Iteration ' + str(repeat + 1)
        output_img = texture.astype(np.uint8)
        cv.imshow(window_name, output_img)

        # apply the the blurred noise texture to the image
        for column in range(columns): # scans across the x-axis
            for row in range(rows): # scans down the y-axis
                for channel in curr_channel:
                    new_pixel = brightness + int(img[row, column, channel] + blend_coeff * texture[row, column])
                    if new_pixel < 255:
                        if new_pixel < 0:
                            img[row, column, channel] = 0
                        else:
                            img[row, column, channel] = new_pixel
                    else:
                        img[row, column, channel] = 255

    # output the final result

    cv.imwrite("problem2_result.jpg", img)
 
    window_name = 'Problem 2 Result'
    output_img = img.astype(np.uint8)
    cv.imshow(window_name, output_img)

    # close all windows when the user presses any key
    cv.waitKey(0)
    cv.destroyAllWindows()



def problem3(img_url, spatial_blur, colour_blur):
    # load the image in
    img = cv.imread(img_url)
    rows,columns,channels = img.shape

    # apply a filter to remove the high-frequency detail
    img = cv.bilateralFilter(img,9,spatial_blur,colour_blur,cv.BORDER_DEFAULT) # recommended values for spatial_blur and colour_blur are 30 and 10

    # output intermediate result
    cv.imwrite("problem3_stage1.jpg", img)
 
    window_name = 'Problem 3 Stage 1'
    output_img = img.astype(np.uint8)
    cv.imshow(window_name, output_img)

    # generate the lookup table using SciPy univariate spline
    # define input data
    y_axis = np.array([0,5,10,20,40,60,80,100,150,200,230,240,245,250,255])
    x_axis = np.array([0,3,6,12,24,36,50,65,110,160,195,210,225,240,255])
    # I flipped the axes so the mapping increases values instead

    LUT = UnivariateSpline(x_axis,y_axis)
    
    # convert the input image back to RGB mode for display
    img = cv.cvtColor(img,cv.COLOR_BGR2HSV)

    for column in range(columns): # scans across the x-axis
        for row in range(rows): # scans down the y-axis
            img[row,column,1] = LUT(img[row,column,1]) # 

    # output the final result

    img = cv.cvtColor(img,cv.COLOR_HSV2BGR)
    
    cv.imwrite("problem3_result.jpg", img)
 
    window_name = 'Problem 3 Result'
    output_img = img.astype(np.uint8)
    cv.imshow(window_name, output_img)

    # close all windows when the user presses any key
    cv.waitKey(0)
    cv.destroyAllWindows()


def problem4(img_url, swirl_angle, swirl_radius, interpolation):

    if interpolation != "nn" and interpolation != "bi":
        print("Error: Invalid interpolation parameter. Assuming nearest neighbour interpolation")
        interpolation = "nn"

    interpolation = "nn" # because my bilinear interpolation doesn't work

    img = cv.imread(img_url)
    rows,columns,channels = img.shape

    # image prefiltering

    # MEDIAN FILTER - this code is based off the median filter in the week 5 practical
    n = 3 # size of filter
    imgNew = np.zeros((rows, columns, channels))
    for row in range(rows): # loop through all rows of output image
        for column in range(columns): # loop through all columns of output image
            for channel in range(channels): # loop through the three colour channels 
                neighbourhood = img[row:row+2*n, column:column+2*n, channel] # define filter neighbourhood 
                imgNew[row,column,channel] = np.median(neighbourhood) # compute median value 

    img = imgNew

    cv.imwrite("problem4_stage_1.jpg", img)
 
    window_name = 'Problem 4 Stage 1'
    output_img = img.astype(np.uint8)
    cv.imshow(window_name, output_img)

    # add check for valid swirl radius
    if swirl_radius > rows/2 or swirl_radius > columns/2:
        print("Error: the swirl radius is larger than the image! Now using maximum swirl radius.")
        swirl_radius = min(rows/2,columns/2)

    centre_x = columns / 2
    centre_y = rows / 2

    swirl_img = np.zeros((columns,rows,channels))

    for x in range(columns): # scans across the x-axis
        for y in range(rows): # scans down the y-axis
            dist_x = x - centre_x # face is at the centre of the image
            dist_y = y - centre_y
            dist = np.sqrt(dist_x**2 + dist_y**2)

            end_angle = np.arctan2(dist_x,dist_y)

            relative_swirl = (swirl_radius-dist)/20
            if relative_swirl < 0:
                relative_swirl = 0

            # find the pixel in the original image that will be swirled to end up here
            start_angle = end_angle + (swirl_angle * relative_swirl)

            start_x = np.sin(start_angle) * dist + centre_x
            start_y = np.cos(start_angle) * dist + centre_y

            if interpolation == "nn":
                start_x = round(start_x)
                start_y = round(start_y)
                swirl_img[x,y,:] = img[start_x,start_y,:] # copy all colour channels
            elif interpolation == "bi":
                left = math.floor(start_x)
                right = math.ceil(start_x)
                top = math.floor(start_y)
                bottom = math.ceil(start_y)
                left_dist = x - left
                right_dist = right - x
                top_dist = y - top
                bottom_dist = bottom - y
                swirl_img[x,y,:] = img[left,top,:]*left_dist*top_dist + img[left,bottom,:]*left_dist*bottom_dist + img[right,bottom,:]*right_dist*bottom_dist + img[right,top,:]*right_dist*top_dist


    cv.imwrite("problem4_result.jpg", swirl_img)
 
    window_name = 'Problem 4 Result'
    output_img = swirl_img.astype(np.uint8)
    cv.imshow(window_name, output_img)

    # close all windows when the user presses any key
    cv.waitKey(0)
    cv.destroyAllWindows()
