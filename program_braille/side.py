
from  OBR import SegmentationEngine,BrailleClassifier,BrailleImage
import cv2
from gtts import gTTS
image_path ='D:\\project\\collage\\braille\\Screenshot 2023-03-23 124740.png'

cv2.imshow("braille Input image",cv2.imread(image_path))
cv2.waitKey(100000)

classifier = BrailleClassifier()

img = BrailleImage(image_path)
for letters in SegmentationEngine(image=img):
    letters.mark()
    classifier.push(letters)
    classifier.digest()

print("{}\n".format(classifier.digest()))


tts = gTTS(classifier.digest(),lang ='en', tld='co.in')

tts.save('hello.mp3')



