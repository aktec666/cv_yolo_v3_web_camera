import cv2
from imageai.Detection import ObjectDetection 

vid = cv2.VideoCapture(0) 
  
detector = ObjectDetection()
model_path = "yolov3.pt"
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

while(True): 
    ret, frame = vid.read()

    scale_percent = 50
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite("frame.jpg", resized) 

    detections = detector.detectObjectsFromImage(
        input_image="frame.JPG", 
        output_image_path="output.jpg", 
        minimum_percentage_probability=30)
    
    img = cv2.imread("output.jpg") 

    cv2.imshow('камера', img) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows() 