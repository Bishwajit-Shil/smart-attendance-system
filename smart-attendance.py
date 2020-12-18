import cv2
import numpy as np 
import face_recognition
import os
import imutils
from datetime import  datetime

# Read images and classNames
path = 'Attendance-img/'
images =[]
classNames =[]
my_list = os.listdir(path)
# print(my_list)
for cl in my_list:
	curImg = cv2.imread(f'{path}/{cl}')
	images.append(curImg)
	classNames.append(os.path.splitext(cl)[0])

# print(classNames)

# Encode images
def findEncodings(images):
    encodingList = []
    for img in images:
    	img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    	encode = face_recognition.face_encodings(img)[0]
    	encodingList.append(encode)
    return encodingList

def  markattendance(name,date):
	with open('Attendance.csv', 'r+') as f:
		myData_list = f.readlines()
		# print(myData_list)
		nameList = []
		dateList = []
		for line in myData_list:
			entry = line.split(',')
			nameList.append(entry[0])
			dateList.append(entry[2])
		print(dateList)	

		if name not in nameList or date not in dateList:
			now  = datetime.now()
			dstring = now.strftime('%H:%M:%S')
			d_date = now.strftime('%Y:%m:%d')
			f.writelines(f'\n{name},{dstring},{d_date}')

encodelist_known = findEncodings(images)
print('Encoding complete')


# face_recognize face name
video = 'test/test.mp4'
cap = cv2.VideoCapture(video)
while True:
	__, img = cap.read()
	# img = cv2.imread('test/1.jpg')
	img = imutils.resize(img, height=300)
	imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	facecur_frame = face_recognition.face_locations(imgs)
	face_encodeframe = face_recognition.face_encodings(imgs, facecur_frame)

	for encodeface, faceloca in zip(face_encodeframe, facecur_frame):
		matches = face_recognition.compare_faces(encodelist_known , encodeface)
		face_dis = face_recognition.face_distance(encodelist_known, encodeface)
		# print(face_dis)
		matchIndex = np.argmin(face_dis)

		if matches[matchIndex]:
			name = classNames[matchIndex].upper()
			# print(name)
			y1,x2,y2,x1  = faceloca
			# print(faceloca)
			cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
			cv2.rectangle(img, (x1, y1-20), (x2,y1), (0,255,0),-1)
			cv2.putText(img, name,  (x1-5, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0,0,0), 2)	
			cv2.imshow('Smart - Attendance',img)

			now  = datetime.now()
			date = now.strftime('%Y:%m:%d')
			markattendance(name,date)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


