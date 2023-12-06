#thêm thư viện
from imageai.Detection import ObjectDetection
import os

#truy cập vào đường dẫn thư mục hiện tại
execution_path = os.getcwd()

#thêm model YOLO V3
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()

#truyền input là hình ảnh cần nhận diện đối tượng 
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , 
"E:\\Nam 3\\TH ML\\Nhom02_YOLOv3\\test-images\\hehe.jpg"), 
output_image_path=os.path.join(execution_path , 
"E:\\Nam 3\\TH ML\\Nhom02_YOLOv3\\test-results\\hehe_result.jpg"), 
minimum_percentage_probability=30)

#in ra tên đối tượng được nhận diện, phần trăm chính xác, các điểm bouding box
for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], 
    " : ", eachObject["box_points"] )
    print("--------------------------------")