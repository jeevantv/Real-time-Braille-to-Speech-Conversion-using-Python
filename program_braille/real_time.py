import cv2

def capture_image():
    # initialize the camera
    cap = cv2.VideoCapture(0)

    # keep capturing frames until 'c' is pressed
    while True:
        ret, frame = cap.read()

        # display the captured frame
        cv2.imshow('frame', frame)

        # wait for 'c' key to be pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    # release the camera and destroy the window
    cap.release()
    cv2.destroyAllWindows()

    return frame

def crop_image(image):
    # display the image and wait for the user to select a region of interest (ROI)
    cv2.imshow('image', image)
    roi = cv2.selectROI('image', image, fromCenter=False, showCrosshair=True)

    # crop the image using the selected ROI
    cropped_image = image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

    # display the cropped image and wait for 'enter' key to be pressed
    cv2.imshow('cropped image', cropped_image)
    cv2.waitKey(0)
    return cropped_image

if __name__ == '__main__':
    # capture the image
    image = capture_image()

    # crop the image
    cropped_image = crop_image(image)
    

class BrailleCharacter(object):
    def __init__(self, dot_coordinates, diameter, radius, parent_image):
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None
        self.dot_coordinates = dot_coordinates
        self.diameter = diameter
        self.radius = radius
        self.parent_image = parent_image
        return;

    def mark(self):
        self.parent_image.bound_box(self.left,self.right,self.top,self.bottom)
        return;

    def get_parent_image(self):
        return self.parent_image

    def get_dot_diameter(self):
        return self.diameter

    def get_dot_radius(self):
        return self.radius

    def get_dot_coordinates(self):
        return self.dot_coordinates

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_top(self):
        return self.top

    def get_bottom(self):
        return self.bottom

    def get_opencv_left_top(self):
        return (self.left, self.top)
        
    def get_opencv_right_bottom(self):
        return (self.right, self.bottom)

    def get_bounding_box(self, form = "left,right,top,bottom"):
        r = []
        form = form.split(',')
        if len(form) < 8:
            return (self.left,self.right,self.top,self.bottom)

        for direction in form:
            direction = direction.lower()
            if direction == 'left':
                r.append(self.left)
            elif direction == 'right':
                r.append(self.right)
            elif direction == 'top':
                r.append(self.top)
            elif direction == 'bottom':
                r.append(self.bottom)
            else:
                return (self.left,self.right,self.top,self.bottom)
        
        return tuple(r)

    def is_valid(self):
        r = True
        r = r and (self.left is not None)
        r = r and (self.right is not None)
        r = r and (self.top is not None)
        r = r and (self.bottom is not None)
        return r

from math import sqrt

def get_distance(p1, p2):
        x1,y1 = p1
        x2,y2 = p2
        return (x2 - x1)**2 + (y2 - y1)**2

def get_left_nearest(dots, diameter, left):
        nearest = None
        for dot in dots:
            x,y = dot[0]
            dist = int(x - left)
            if dist <= diameter:
                if nearest is None:
                    nearest = dot
                else:
                    X,Y = nearest[0]
                    DIST = int(X - left)
                    if DIST > dist:
                        nearest = dot
        return nearest

def get_right_nearest(dots, diameter, right):
        nearest = None
        for dot in dots:
            x,y = dot[0]
            dist = int(right - x)
            if dist <= diameter:
                if nearest is None:
                    nearest = dot
                else:
                    X,Y = nearest[0]
                    DIST = int(right - X)
                    if DIST > dist:
                        nearest = dot
        return nearest

def get_dot_nearest(dots, diameter, pt1):
        nearest = None
        diameter **= 2
        for dot in dots:
            point = dot[0]
            dist_from_pt1 = get_distance(point, pt1)
            if dist_from_pt1 <= diameter:
                if nearest is None:
                    nearest = dot
                else:
                    pt = nearest[0]
                    ndist_from_pt1 = get_distance(pt, pt1)
                    if ndist_from_pt1 >= dist_from_pt1:
                        nearest = dot
        return nearest



def get_combination(box, dots, diameter):
        result = [0,0,0,0,0,0]
        left,right,top,bottom = box

        midpointY = int((bottom - top)/2)
        end = (right, midpointY)
        start = (left, midpointY)
        width = int(right - left)

        corners = { (left,top): 1, (right,top): 4, (left, bottom): 3, (right,bottom): 6,
                (left): 2, (right): 5}

        for corner in corners:
            if corner != left and corner != right:
                D = get_dot_nearest(dots, int(diameter), corner)
            else:
                if corner == left:
                    D = get_left_nearest(dots, int(diameter), left)
                else:
                    D = get_right_nearest(dots, int(diameter), right)

            if D is not None:
                dots.remove(D)
                result[corners[corner]-1] = 1

            if len(dots) == 0:
                break
        return end,start,width,tuple(result);

def translate_to_number(value):
    if value == 'a':
        return '1'
    elif value == 'b':
        return '2'
    elif value == 'c':
        return '3'
    elif value == 'd':
        return '4'
    elif value == 'e':
        return '5'
    elif value == 'f':
        return '6'
    elif value == 'g':
        return '7'
    elif value == 'h':
        return '8'
    elif value == 'i':
        return '9'
    else:
        return '0'

class Symbol(object):
    def __init__(self, value = None, letter = False, special = False):
        self.is_letter = letter
        self.is_special = special
        self.value = value

    def is_valid(self):
        r = True
        r = r and (self.value is not None)
        r = r and (self.is_letter is not None or self.is_special is not None)
        return r

    def letter(self):
        return self.is_letter

    def special(self):
        return self.is_special

class BrailleClassifier(object):
    symbol_table = {
         (1,0,0,0,0,0): Symbol('a',letter=True),
         (1,1,0,0,0,0): Symbol('b',letter=True),
         (1,0,0,1,0,0): Symbol('c',letter=True),
         (1,0,0,1,1,0): Symbol('d',letter=True),
         (1,0,0,0,1,0): Symbol('e',letter=True),
         (1,1,0,1,0,0): Symbol('f',letter=True),
         (1,1,0,1,1,0): Symbol('g',letter=True),
         (1,1,0,0,1,0): Symbol('h',letter=True),
         (0,1,0,1,0,0): Symbol('i',letter=True),
         (0,1,0,1,1,0): Symbol('j',letter=True),
         (1,0,1,0,0,0): Symbol('K',letter=True),
         (1,1,1,0,0,0): Symbol('l',letter=True),
         (1,0,1,1,0,0): Symbol('m',letter=True),
         (1,0,1,1,1,0): Symbol('n',letter=True),
         (1,0,1,0,1,0): Symbol('o',letter=True),
         (1,1,1,1,0,0): Symbol('p',letter=True),
         (1,1,1,1,1,0): Symbol('q',letter=True),
         (1,1,1,0,1,0): Symbol('r',letter=True),
         (0,1,1,1,0,0): Symbol('s',letter=True),
         (0,1,1,1,1,0): Symbol('t',letter=True),
         (1,0,1,0,0,1): Symbol('u',letter=True),
         (1,1,1,0,0,1): Symbol('v',letter=True),
         (0,1,0,1,1,1): Symbol('w',letter=True),
         (1,0,1,1,0,1): Symbol('x',letter=True),
         (1,0,1,1,1,1): Symbol('y',letter=True),
         (1,0,1,0,1,1): Symbol('z',letter=True),
         (0,0,1,1,1,1): Symbol('#',special=True),
    }
    def __init__(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False
        return;

    def push(self, character):
        if not character.is_valid():
            return;
        box = character.get_bounding_box()
        dots = character.get_dot_coordinates()
        diameter = character.get_dot_diameter()
        end,start,width,combination = get_combination(box, dots, diameter)
        print('combination',combination)

        if combination not in self.symbol_table:
            self.result += "*"
            return;

        if self.prev_end is not None:
            dist = get_distance(self.prev_end, start)
            if dist*0.5 > (width**2):
                self.result += " "
        self.prev_end = end

        symbol = self.symbol_table[combination]
        if symbol.letter() and self.number:
            self.number = False
            self.result += translate_to_number(symbol.value)
        elif symbol.letter():
            if self.shift_on:
                self.result += symbol.value.upper()
            else:
                self.result += symbol.value
        else:
            if symbol.value == '#':
                self.number = True
        return;

    def digest(self):
        return self.result

    def clear(self):
        self.result = ''
        self.shift_on = False
        self.prev_end = None
        self.number = False
        return;

import cv2
import numpy as np

class BrailleImage(object): 
    def __init__(self, image):
        # Read source image
        self.original =cropped_image
        if self.original is None:
            raise IOError('Cannot open given image')

        # First Layer, Convert BGR(Blue Green Red Scale) to Gray Scale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
        # Save the binary image of the edge detected 
        self.edged_binary_image = self.__get_edged_binary_image(gray)

        # Now do the same to save a binary image to get the contents
        # inside the edges to see if the dot is really filled.
        self.binary_image = self.__get_binary_image(gray)
        self.final = self.original.copy()
        self.height, self.width, self.channels = self.original.shape
        return;

    def bound_box(self,left,right,top,bottom,color= (255,0,0), size=1):
        self.final = cv2.rectangle(self.final, (left, top), (right, bottom), color, size)
        return True

    def get_final_image(self):
        return self.final

    def get_original_image(self):
        return self.original

    def get_edged_binary_image(self):
        return self.edged_binary_image

    def get_binary_image(self):
        return self.binary_image

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def __get_edged_binary_image(self, gray):
        # First Lvl Blur to Reduce Noise
        blur = cv2.GaussianBlur(gray,(3,3),0)
        # Adaptive Thresholding to define the  dots in Braille
        thres = cv2.adaptiveThreshold(
                blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,57,5) 
        # Remove more Noise from the edges.
        blur2 = cv2.medianBlur(thres,3)
        # Sharpen again.
        ret2,th2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Remove more Noise.
        blur3 = cv2.GaussianBlur(th2,(3,3),0)
        # Final threshold
        ret3,th3 = cv2.threshold(blur3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (-1, -1))
        dilated = cv2.dilate(th3, element)
        eroded = cv2.erode(dilated, element)
        return cv2.bitwise_not(eroded)

    def __get_binary_image(self, gray):
        blur     = cv2.GaussianBlur(gray,(3,3),0)
        ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        blur2    = cv2.medianBlur(th2,3)
        ret3,th3 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (-1, -1))
        dilated = cv2.dilate(th3, element)
        eroded = cv2.erode(dilated, element)
        return cv2.bitwise_not(eroded)

import cv2
import numpy as np
from math import sqrt
from collections import Counter

class SegmentationEngine(object): 
    def __init__(self, image = None):
        self.image = image
        self.initialized = False
        self.dots = []
        self.diameter = 0.0
        self.radius = 0.0
        self.next_epoch = 0
        self.characters = []
        return;

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if not self.initialized:
            self.initialized = True
            contours = self.__process_contours()
            if len(contours) == 0:
                # Since we have no dots.
                self.__clear()
                raise StopIteration()
            enclosingCircles = self.__get_min_enclosing_circles(contours)
            if len(enclosingCircles) == 0:
                self.__clear()
                raise StopIteration()

            diameter,dots,radius = self.__get_valid_dots(enclosingCircles)
            if len(dots) == 0:
                self.__clear()
                raise StopIteration()
            self.diameter = diameter
            self.dots = dots
            self.radius = radius
            self.next_epoch = 0
            self.characters = []

        if len(self.characters) > 0:
            r = self.characters[0]
            del self.characters[0]
            return r

        cor = self.__get_row_cor(self.dots, epoch=self.next_epoch) # do not respect breakpoints
        if cor is None:
            self.__clear()
            raise StopIteration()

        top = int(cor[1] - int(self.radius*1.5)) # y coordinate
        self.next_epoch = int(cor[1] + self.radius)

        cor = self.__get_row_cor(self.dots,self.next_epoch,self.diameter,True)
        if cor is None:
            # Assume next epoch
            self.next_epoch = int(self.next_epoch + (2*self.diameter))
        else:
            self.next_epoch = int(cor[1] + self.radius)

        cor = self.__get_row_cor(self.dots,self.next_epoch,self.diameter,True)
        if cor is None:
            self.next_epoch = int(self.next_epoch + (2*self.diameter))
        else:
            self.next_epoch = int(cor[1] + self.radius)
        
        bottom = self.next_epoch
        self.next_epoch += int(2*self.diameter)

        DOI = self.__get_dots_from_region(self.dots, top, bottom)
        xnextEpoch = 0
        while True:
            xcor = self.__get_col_cor(DOI, xnextEpoch)
            if xcor is None:
                break

            left = int(xcor[0] - self.radius) # x coordinate
            xnextEpoch = int(xcor[0] + self.radius)
            xcor = self.__get_col_cor(DOI,xnextEpoch,self.diameter,True)
            if xcor is None:
                # Assumed
                xnextEpoch += int(self.diameter*1.5)
            else:
                xnextEpoch = int(xcor[0]) + int(self.radius)
            right = xnextEpoch
            box = (left, right, top, bottom)
            dts = self.__get_dots_from_box(DOI, box)
            char = BrailleCharacter(dts, self.diameter, self.radius, self.image)
            char.left = left
            char.right = right
            char.top = top
            char.bottom = bottom
            self.characters.append(char)

        if len(self.characters) < 1:
            self.__clear()
            raise StopIteration()

        r = self.characters[0]
        del self.characters[0]
        return r

    def __clear(self):
        self.image = None
        self.initialized = False
        self.dots = []
        self.diameter = 0.0
        self.radius = 0.0
        self.next_epoch = 0
        self.characters = []

    def update(self, image):
        self.__clear()
        self.image = image
        return True

    def __get_row_cor(self, dots, epoch = 0, diameter = 0, respectBreakpoint = False):
        if len(dots) == 0:
            return None
        minDot = None
        for dot in dots:
            x,y = dot[0]
            if y < epoch:
                continue

            if minDot is None:
                minDot = dot
            else:
                v = int(y - epoch)
                minV = int(minDot[0][1] - epoch)
                if minV > v:
                    minDot = dot
                else:
                    continue
        if minDot is None:
            return None
        if respectBreakpoint:
            v = int(minDot[0][1] - epoch)
            if v > (2*diameter):
                return None # indicates that the entire row is not set
        return minDot[0] # (X,Y)

    def __get_col_cor(self, dots, epoch = 0, diameter = 0, respectBreakpoint = False):
        if len(dots) == 0:
            return None
        minDot = None
        for dot in dots:
            x,y = dot[0]
            if x < epoch:
                continue

            if minDot is None:
                minDot = dot
            else:
                v = int(x - epoch)
                minV = int(minDot[0][0] - epoch)
                if minV > v:
                    minDot = dot
                else:
                    continue
        if minDot is None:
            return None
        if respectBreakpoint:
            v = int(minDot[0][0] - epoch)
            if v > (2*diameter):
                return None # indicates that the entire row is not set
        return minDot[0] # (X,Y)

    def __get_dots_from_box(self, dots, box):
        left,right,top,bottom = box
        result = []
        for dot in dots:
            x,y = dot[0]
            if x >= left and x <= right and y >= top and y <= bottom:
                result.append(dot)
        return result

    def __get_dots_from_region(self, dots, y1, y2):
        D = []
        if y2 < y1:
            return D

        for dot in dots:
            x,y = dot[0]
            if y > y1 and y < y2:
                D.append(dot)
        return D

    def __get_valid_dots(self, circles):
        tolerance = 0.45
        radii = []
        consider = []
        bin_img = self.image.get_binary_image()
        for circle in circles:
            x,y = circle[0]
            rad = circle[1]
            # OpenCV uses row major
            # Since we do a bitwise not, white pixels belong to the dot.
            
            # Go through the x axis and check if all those are white
            # pixels till you reach the rad
            it = 0
            while it < int(rad):
                if bin_img[y,x+it] > 0 and bin_img[y+it,x] > 0:
                    it += 1
                else:
                    break
            else:
                if bin_img[y,x] > 0:
                    consider.append(circle)
                    radii.append(rad)

        baserad = Counter(radii).most_common(1)[0][0]
        dots = []
        for circle in consider:
            x,y = circle[0]
            rad = circle[1]
            if rad <= int(baserad * (1+tolerance)) and rad >= int(baserad * (1-tolerance)):
                dots.append(circle)

        # Remove duplicate enclosing circles
        # (i.e) Remove circle enclosed by another other circle.
        for dot in dots:
            X1,Y1 = dot[0]
            C1 = dot[1]
            for sdot in dots:
                if dot == sdot:
                    continue
                X2,Y2 = sdot[0]
                C2 = sdot[1]
                D = sqrt(((X2 - X1)**2) + ((Y2-Y1)**2))
                if C1 > (D + C2):
                    dots.remove(sdot)
        
        # Filtered base radius
        radii = []
        for dot in dots:
            rad = dot[1]
            radii.append(rad)
        baserad = Counter(radii).most_common(1)[0][0] 
        return 2*(baserad), dots, baserad
            
    def __get_min_enclosing_circles(self, contours):
        circles = []
        radii = []
        for contour in contours:
            (x,y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            radii.append(radius)
            circles.append((center, radius))
        return circles

    def __process_contours(self):
        edg_bin_img = self.image.get_edged_binary_image()
        contours = cv2.findContours(edg_bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 2:
            contours = contours[0]
        else:
            contours = contours[1]
        return contours

import cv2

classifier = BrailleClassifier()        
image_path =cropped_image
img = BrailleImage(image_path)
for letter in SegmentationEngine(image=img):
    letter.mark()
    classifier.push(letter)
cv2.imshow('1', img.get_final_image())
cv2.imshow('2',img.get_binary_image())

cv2.waitKey(0)

classifier = BrailleClassifier()
path=cropped_image
img = BrailleImage(path)
for letter in SegmentationEngine(image=img):
    classifier.push(letter)
    classifier.digest()
    mytext=classifier.digest()
print("{}: {}\n", classifier.digest())
cv2.waitKey(0)
from gtts import gTTS

language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save('hello.mp3')
print(mytext)










