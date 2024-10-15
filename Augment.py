import numpy as np
import cv2
import random
import glob
import math

# choose classess(images) names
classnames = ['stop.png','straight.png','left.png','right.png','giveWay.png','tunnel.png','tunnel end.PNG','pass.png','end pass.png','crosswalk.png','downhill.png','uphill.png','priority.png','park.png']
classnamesBW = ['stop BW.png','straight BW.png','left BW.png','right BW.png','giveWay BW.png','tunnel BW.png','tunnel end BW.PNG','pass BW.png','end pass BW.png','crosswalk BW.png','downhill BW.png','uphill BW.png','priority BW.png','park BW.png']


def draw_bounding_box_and_crop(img , imgColor):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold image to make it black and white
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # find contours of white shapes in image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find largest white shape
    largest_contour = max(contours, key=cv2.contourArea)

    # get bounding box of largest white shape
    x, y, w, h = cv2.boundingRect(largest_contour)

    # draw bounding box on image
    bbox_img = cv2.rectangle(np.copy(img), (x, y), (x + w, y + h), (255, 255, 255), 2)

    # crop image inside bounding box
    cropped_img = img[y:y+h, x:x+w]
    imgColor = imgColor[y:y+h, x:x+w]
    return imgColor, cropped_img

def four_point_transform(im, ran, width, height):
    # obtain a consistent order of the points and unpack them individually
    randomChoice = ran[0]
    ran = ran[1]
    if randomChoice == 1:
        pts = np.array([[0,im.shape[0]//ran],[im.shape[1]-1,0],[im.shape[1]-1,im.shape[0]-1],[0,im.shape[0]*(ran-1)//ran]])
    elif randomChoice == 2:
        pts = np.array([[im.shape[1]//ran,0],[im.shape[1]*(ran-1)//ran,0],[im.shape[1]-1,im.shape[0]-1],[0,im.shape[0]-1]])
    elif randomChoice == 3:
        pts = np.array([[0,0],[im.shape[1]-1,im.shape[0]//ran],[im.shape[1]-1,im.shape[0]*(ran-1)//ran],[0,im.shape[0]-1]])
    elif randomChoice == 4:
        pts = np.array([[0,0],[im.shape[1]-1,0],[im.shape[1]*(ran-1)//ran,im.shape[0]-1],[im.shape[1]//ran,im.shape[0]-1]])
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # calculate the width and height of the new image
    (tl, tr, br, bl) = rect
    maxWidth = max(int(np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))),
                   int(np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))))
    maxHeight = max(int(np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))),
                    int(np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))))

    # construct the set of destination points to obtain warped view (i.e. top-down view) of the image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(dst,rect)
    warped = cv2.warpPerspective(im, M, (maxWidth, maxHeight))

    # resize the warped image to the specified width and height
    warped = cv2.resize(warped, (width, height))

    # return the warped image
    return warped

def add_random_rectangle(input_image):
    # get image dimensions
    height, width, channels = input_image.shape

    # calculate maximum rectangle area
    max_area = int(0.25 * width * height)

    # generate random rectangle size and position
    rect_area = random.randint(0, max_area)
    rect_aspect_ratio = random.uniform(0.4, 2.5)
    rect_height = int(np.sqrt(rect_area / rect_aspect_ratio))
    rect_width = int(rect_aspect_ratio * rect_height)
    rect_left = random.randint(0, width - rect_width)
    rect_top = random.randint(0, height - rect_height)

    # create black rectangle image
    rect_img = np.zeros((height, width, channels), dtype=np.uint8)
    cv2.rectangle(rect_img, (rect_left, rect_top), (rect_left + rect_width, rect_top + rect_height), (0, 0, 0), -1)

    # mask out random rectangle area of input image
    masked_image = np.copy(input_image)
    masked_image[rect_top:rect_top+rect_height, rect_left:rect_left+rect_width, :] = 0

    # combine original and black rectangle images
    result = cv2.add(masked_image, rect_img)

    return result

def add_random_blur(im):
    # Generate a random kernel size (odd integer between 3 and 11)
    kernel_size = np.random.choice(range(3, 13, 2))

    # Generate a random standard deviation for Gaussian distribution
    std_dev = np.random.uniform(0, 3)

    # Apply Gaussian blur to the input image
    blurred_image = cv2.GaussianBlur(im, (kernel_size, kernel_size), std_dev)

    return blurred_image

def black_and_white(im):
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return im

def salt_and_pepper(im):
    noisy_image = np.copy(im)

    # Determine the number of pixels to be affected by the noise (random amount between 0 and 30 percent)
    amount = np.random.uniform(0, 0.3)
    num_pixels = int(amount * im.shape[0] * im.shape[1])

    # Randomly select pixel locations to add noise to
    indices = np.random.choice(range(im.shape[0] * im.shape[1]), size=num_pixels, replace=False)

    # Split the image into its three color channels (BGR)
    b, g, r = cv2.split(noisy_image)

    # Add salt and pepper noise to the selected pixel locations in each color channel
    b[np.unravel_index(indices, im.shape[:2])] = np.random.choice([0, 255], size=(num_pixels,))
    g[np.unravel_index(indices, im.shape[:2])] = np.random.choice([0, 255], size=(num_pixels,))
    r[np.unravel_index(indices, im.shape[:2])] = np.random.choice([0, 255], size=(num_pixels,))

    # Merge the color channels back into a single image
    noisy_image = cv2.merge((b, g, r))

    return noisy_image

def randomRotate():
    random_float = random.uniform(-35, 35)
    random_float = random_float * random.gauss(0, 1)
    random_int = int(random_float)
    return random_int

def rotate_image(im, angle):
    rows, cols = im.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    result = cv2.warpAffine(im, rotation_matrix, (cols, rows))
    return result

def resize(im,imBW):
    max_length = min(back.shape[0],back.shape[1])//2
    min_length  = max_length//9
    size = random.randint(min_length,max_length)
    # print(min_length,max_length,size)
    im = cv2.resize(im , (size,size))
    imBW = cv2.resize(imBW , (size,size))
    return im,imBW

def limits():
    if(image.shape[1]%2 == 0):
        x_min = int(image.shape[1]*1.1)//2
        x_max = back.shape[1]-(int(image.shape[1]*1.1)//2)
    else:
        x_min = int(image.shape[1]*1.1)//2
        x_max = back.shape[1]-(int(image.shape[1]*1.1)//2) + 1
    if(image.shape[0]%2 == 0):
        y_min = int(image.shape[0]*1.1)//2
        y_max = back.shape[0] - int(image.shape[0]*1.1)//2
    else:
        y_min = int(image.shape[0]*1.1)//2
        y_max = back.shape[0] - int(image.shape[0]*1.1)//2 + 1

    return x_min,x_max,y_min,y_max


def center():
    x_min ,x_max ,y_min ,y_max = limits()
    x_center = random.randint(x_min,x_max)
    y_center = random.randint(y_min,y_max)
    return x_center,y_center

def add(back , img, imgBW):
    imageCenter = center()
    # print(img.shape)
    # print(imageCenter)
    if(img.shape[1]%2 == 0):
        x_min = int(imageCenter[0])-int(img.shape[1]//2)
        x_max = int(imageCenter[0])+int(img.shape[1]//2)
    else:
        x_min = int(imageCenter[0])-int(img.shape[1]//2)
        x_max = int(imageCenter[0])+int(img.shape[1]//2) + 1
    if(img.shape[0]%2 == 0):
        y_min = int(imageCenter[1])-int(img.shape[0]//2)
        y_max = int(imageCenter[1])+int(img.shape[0]//2)
    else:
        y_min = int(imageCenter[1])-int(img.shape[0]//2)
        y_max = int(imageCenter[1])+int(img.shape[0]//2) + 1
    
    
    print(imageCenter,back.shape,img.shape,imgBW.shape,y_min,y_max,x_min,x_max)
    im = imgBW[:,:,0]
    im[im > 40 ] = 100
    imgBW[im <= 40] = back[y_min:y_max,x_min:x_max][im <= 40]
    imgBW[im == 100] = img[im == 100]

    back[y_min:y_max,x_min:x_max] = imgBW

    cx = round(imageCenter[0]/back.shape[1],5)
    cy = round(imageCenter[1]/back.shape[0],5)
    b = round(img.shape[1]/back.shape[1],5)
    a = round(img.shape[0]/back.shape[0],5)
    
    return back,cx,cy,b,a

i = 0
indx = 0
# for indx in range(len(classnames)):
for filename in glob.glob('*.jpg'):
    # try:
    print(indx,classnames[indx],len(classnames),i)
    image = cv2.imread(classnames[indx])
    imageBW = cv2.imread(classnamesBW[indx])
    back = cv2.imread(filename)
    image , imageBW = resize(image,imageBW)


    randomNum = random.randint(1,10)
    if randomNum <= 2:
        image , imageBW=draw_bounding_box_and_crop(imageBW,image)
        imageBW = add_random_rectangle(imageBW)
    elif randomNum >= 3 and randomNum <=4:
        warpRand = np.random.choice(range(5, 11, 2))
        randomChoice = random.randint(1,4)
        imageBW = four_point_transform(imageBW,(randomChoice,warpRand),imageBW.shape[1],imageBW.shape[0])
        image = four_point_transform(image,(randomChoice,warpRand),image.shape[1],image.shape[0])
        image , imageBW=draw_bounding_box_and_crop(imageBW,image)
    elif randomNum >= 5 and randomNum <=10:
        rotateDegree = randomRotate()
        imageBW = rotate_image(imageBW, rotateDegree)
        image = rotate_image(image, rotateDegree)
        image , imageBW=draw_bounding_box_and_crop(imageBW,image)

    augimg,yoloa,yolob,yoloc,yolod = add(back,image,imageBW)
    
    randomNum = random.randint(1,10)
    if randomNum >= 6 and randomNum <=7:
        augimg = salt_and_pepper(augimg)
    if randomNum >= 7 and randomNum <=8:
        augimg = add_random_blur(augimg)
    if randomNum >= 8 and randomNum <=10:
        augimg = black_and_white(augimg)
    
    max_length = max(augimg.shape[0],augimg.shape[1])
    ratio = max_length/640
    augimg = cv2.resize(augimg , (int(augimg.shape[1]//ratio),int(augimg.shape[0]//ratio)))

    #show
    # cv2.imshow("added",augimg)
    # cv2.waitKey(0)

    #save
    name = str("aug" + str(i) + ".jpg")
    cv2.imwrite(name,augimg)
    yoloname = str(str(indx) + " " + str(yoloa) + " " + str(yolob) + " " + str(yoloc) + " " + str(yolod))
    with open(str("aug" + str(i)+".txt"), 'w') as f:
        f.write(yoloname)
    if i % 10 == 0 and i >= 1 and (i//10 -1) == indx :
        indx +=1
        # break
    i += 1
