import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import ImageTk, Image

from imageai.Detection import ObjectDetection
import os

class GUI:
    def __init__(self, root):
        self.file_path = ''
        self.default_path = 'E:\\Nam 3\\TH ML\\Nhom02\\Nhom02_YOLOv3\\test-results\\detect_result.jpg'

        self.root = root
        self.root.title("GUI with tkinter")
        self.root.geometry("800x600")

        self.btn_select_image = tk.Button(root, text="Select Image", command=self.select_image)
        self.btn_select_image.pack()

        # Tạo khung văn bản để hiển thị đoạn văn bản
        self.text_frame = ScrolledText(root, height=10, width=40)
        self.text_frame.pack()

        # Tạo nút xử lý
        self.btn_process = tk.Button(root, text="Process", command=self.process)
        self.btn_process.pack()

        # Tạo nút chọn tệp ảnh và hiển thị hình ảnh
        self.image_frame = tk.Label(root)
        self.image_frame.pack()

    def select_image(self):
        # Hiển thị hộp thoại chọn tệp ảnh
        self.file_path = filedialog.askopenfilename()
        self.text_frame.delete('1.0', tk.END)  # Xóa nội dung cũ
        self.text_frame.insert(tk.INSERT, self.file_path)  # Chèn nội dung mới

    def process(self):
        execution_path = os.getcwd()

        detector = ObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(
            input_image=self.file_path,
            output_image_path=self.default_path,
            minimum_percentage_probability=30
        )
        
        # Print results
        self.text_frame.delete('1.0', tk.END)  # Xóa nội dung cũ
        for eachObject in detections:
            info = f'{eachObject["name"]} : {eachObject["percentage_probability"]} : {eachObject["box_points"]}\n'
            self.text_frame.insert(tk.INSERT, info)

        # Hiển thị ảnh
        image = Image.open(self.default_path)
        image = image.resize((600, 380), Image.LANCZOS)  # Thay đổi kích thước ảnh
        photo = ImageTk.PhotoImage(image)

        self.image_frame.config(image=photo)
        self.image_frame.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()