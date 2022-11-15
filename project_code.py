import numpy as np
import cv2

# Take a look at our digits dataset
image = cv2.imread('./data/dataset.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)
cv2.imshow('Digits Image', small)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# just adjust some light
for i in range(37,56,1):
    for j in range(120):
        if(gray[i][j] >= 200):
            gray[i][j] += 15
for i in range(56,79,1):
    for j in range(120):
        if(gray[i][j] >= 190):
            gray[i][j] += 22
for i in range(79,100,1):
    for j in range(120):
        if(gray[i][j] >= 180):
            gray[i][j] += 35
for i in range(100,120,1):
    for j in range(120):
        if(gray[i][j] >= 170):
            gray[i][j] += 45
for i in range(120,139,1):
    for j in range(120):
        if(gray[i][j] >= 160):
            gray[i][j] += 55
for i in range(139,160,1):
    for j in range(120):
        if(gray[i][j] >= 150):
            gray[i][j] += 65


# Blur image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("blurred", blurred)

ret , train_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

# Split the image to 48 cells, each 20x20 size
# This gives us a 4-dim array: 8 x 6 x 20 x 20
cells = [np.hsplit(row,6) for row in np.vsplit(train_image,8)]

# Convert the List data type to Numpy Array of shape (8,6,20,20)
x = np.array(cells)
print ("The shape of our cells array: " + str(x.shape))

train = x[:,:6].reshape(-1,20*20).astype(np.float32) # Size = (3500,400)

# Create labels for train data
k = [0,1,2,3,10,11,12,13]
train_labels = np.repeat(k,6)[:,np.newaxis]

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE ,train_labels)

#四則運算功能
def Arithmetic(original_number):
    #要先creat兩個list
    value = []
    symbol = []
    #用來記得加減乘除位置的變數
    temp = []
    seperate_symbol(original_number,symbol,temp)
    seperate_value(original_number,value,temp)
    multiDiv_operation(value,symbol)
    addMin_operation(value,symbol)
    return value

#抓出加減乘除符號
def seperate_symbol(original_number,symbol,temp):
    for i in range(len(original_number)):
        if(original_number[i]>=10):
            symbol.append(original_number[i])
            temp.append(i)
            
#算出影像數字的實際數值
def indeed_value(seperate):
    total = 0
    for i in range(len(seperate)):
        total += seperate[i]*pow(10,len(seperate)-i-1)
    return total
    
#抓出各個數字
def seperate_value(original_number,value,temp):
    for i in range(len(temp)):
        if(i==0):
            answer = indeed_value(original_number[:temp[i]])
            value.append(answer)
        else:
            answer = indeed_value(original_number[temp[i-1]+1:temp[i]])
            value.append(answer)
            
            
    lastNumber = indeed_value(original_number[temp[len(temp)-1]+1:len(original_number)])      
    value.append(lastNumber)
        
#先算完乘除部分 
def multiDiv_operation(value,symbol):
    count = 0
    while (count<len(symbol)):
        if(symbol[count] == 12):
            total = value[count] * value[count+1]
            value.pop(count)
            value.pop(count)
            value.insert(count,total)
            symbol.pop(count)
        elif(symbol[count] == 13):
            total = value[count] / float(value[count+1])
            value.pop(count)
            value.pop(count)
            value.insert(count,total)
            symbol.pop(count)
        else:
            count += 1
                          
#加減法
def addMin_operation(value,symbol):
    count = 0
    while (count<len(symbol)):
        if(symbol[count] == 10):
            total = value[count] + value[count+1]
            value.pop(count)
            value.pop(count)
            value.insert(count,total)
            symbol.pop(count)
        elif(symbol[count] == 11):
            total = value[count] - value[count+1]
            value.pop(count)
            value.pop(count)
            value.insert(count,total)
            symbol.pop(count)
        else:
            count += 1

def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return (int(M['m10']/M['m00']))

def makeSquare(not_square):
    # This function takes an image and makes the dimenions square
    # It adds black pixels as the padding where needed
    
    BLACK = [0,0,0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Height = ", height, "Width = ", width)
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        #print("New Height = ", height, "New Width = ", width)
        if (height > width):
            pad = (height - width)/2
            pad = int(pad)
            print(pad)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,0,0,pad,\
                                                   pad,cv2.BORDER_CONSTANT,value=BLACK)
        else:
            pad = (width - height)/2
            pad = int(pad)
            #print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize,pad,pad,0,0,\
                                                   cv2.BORDER_CONSTANT,value=BLACK)
    doublesize_square_dim = doublesize_square.shape
    #print("Sq Height = ", doublesize_square_dim[0], "Sq Width = ", doublesize_square_dim[1])
    return doublesize_square


def resize_to_pixel(dimensions, image):
    # This function then re-sizes an image to the specificied dimenions
    
    buffer_pix = 4
    dimensions  = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0,0,0]
    if (height_r > width_r):
        resized = cv2.copyMakeBorder(resized,0,0,0,1,cv2.BORDER_CONSTANT,value=BLACK)
    if (height_r < width_r):
        resized = cv2.copyMakeBorder(resized,1,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized,p,p,p,p,cv2.BORDER_CONSTANT,value=BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    #print("Padded Height = ", height, "Width = ", width)
    return ReSizedImg

image = cv2.imread('./data/4.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image", image)
cv2.imshow("gray", gray)
cv2.waitKey(0)


# Blur image then find edges using Canny 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)

edged = cv2.Canny(blurred, 30, 150)
cv2.imshow("edged", edged)
cv2.waitKey(0)

# Fint Contours
#edged.copy()是為了不改變原本的圖
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Sort out contours left to right by using their x cordinates
contours = sorted(contours, key = x_cord_contour, reverse = False)

# Create empty array to store entire number
full_number = []
original_number = []

# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)    
    
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    #cv2.imshow("Contours", image)

    if w >= 1 and h >= 5:
        roi = blurred[y:y + h, x:x + w]
        ret, roi = cv2.threshold(roi, 190, 255,cv2.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(20, squared)
        cv2.imshow("final", final)
        final_array = final.reshape((1,20*20))
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
        number = str(int(float(result[0])))
        print(number)
        print('\n')
        full_number.append(number)
        original_number.append(result[0])
        # draw a rectangle around the digit, the show what the
        # digit was classified as
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if(int(number) < 10):
            cv2.putText(image, number, (x , y + 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
        else:
            cv2.putText(image, number, (x , y + 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0) 
        
        
SUM = Arithmetic(original_number)
strSUM = str(SUM[0])
text = 'Total = '+ strSUM
cv2.putText(image, text, (50,500),
            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
cv2.imshow("image", image)
        
#cv2.putText(image, strSUM, (image.shape[1]/2,image.shape[0]*5/6), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2 )
cv2.waitKey(0) 
cv2.destroyAllWindows()
print ("The number is: " + ''.join(full_number))
print("The SUM is ", SUM[0])   