from roboflow import Roboflow
rf = Roboflow(api_key="sOMCjI9UQnfkj760Nyaz")
project = rf.workspace().project("roblab3-detect-human")
model = project.version(1).model
import cv2
video = cv2.VideoCapture(0)


while 1:
# On "q" keypress, exit
 ret, img = video.read()
 detection_results = model.predict(img, confidence=40, overlap=30).json()
 print(detection_results['predictions'])
 for result in detection_results['predictions']:
	  print(result['x'])
	  label = result['class']
	  confidence = result['confidence']
	  x, y, w, h = result['x'],result['y'],result['width'],result['height']
	  x = x - w/2
	  y = y - h/2 

			# Draw the bounding box rectangle
	  cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

			# Draw the label and confidence score
	  text = f"{label}: {confidence:.2f}"
	  cv2.putText(img, text, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

		# Show the frame with detections
 cv2.imshow("Frame", img)
 
 
 if(cv2.waitKey(1) == ord('q')):
   break

# And display the inference results
cap.release()
cv2.destroyAllWindows()
