import os
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter
from ultralytics import YOLO
import cv2
import cvzone
import sys
import math

model = YOLO("/Users/monkey/Public/Python/Diploma_1/Yolo_models/best_filter.pt")
classNames = ['angle', 'fracture', 'line', 'messed_up_angle']


class FileUploadApp:
    def __init__(self, root):
        self.root = root
        self.original_image = None
        self.filtered_image = None
        self.file_path = None

        # Set up the background image
        self.bg_image = Image.open("../Proj_Setup/bg.jpg")
        self.bg_image = self.bg_image.resize((800, 600), Image.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        self.bg_label = Label(self.root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0)

        # create a label
        self.prog_name = Label(root, text="Fracture detection", font=("PT Serif", 24, "bold"), fg="white")
        self.prog_name.pack(pady=5)

        # create button to open file upload window
        self.upload_button = Button(root, text="Upload Image", command=lambda: app.open_upload_window())
        self.upload_button.place(x=360, y=560)

        # create a original image label
        self.prog_name = Label(root, text="\tOriginal image", font=("PT Serif", 12, "bold"), fg="white")
        self.prog_name.place(x=10, y=260)
        # create label to display original image
        self.original_image_label = Label(root)
        self.original_image_label.place(x=10, y=50)

        # create a filtered image label
        self.prog_name = Label(root, text="\tFiltered image", font=("PT Serif", 12, "bold"), fg="white")
        self.prog_name.place(x=10, y=500)
        # create button to show filtered image
        self.filtered_button = Button(root, text="Show Filtered", command=lambda: app.show_filtered_image())
        self.filtered_button.place(x=475, y=560)
        # create label to display filtered image
        self.filtered_image_label = Label(root)
        self.filtered_image_label.place(x=10, y=290)

        # create a filtered image label
        self.prog_name = Label(root, text="\t\t\tResult image", font=("PT Serif", 12, "bold"), fg="white")
        self.prog_name.place(x=250, y=500)
        # create button to show result image
        self.result_button = Button(root, text="Show Result", command=lambda: app.show_result_image())
        self.result_button.place(x=590, y=560)
        # create label to display filtered image
        self.result_image_label = Label(root)
        self.result_image_label.place(x=250, y=50)

        # create logout button
        self.logout_button = Button(root, text="Logout", command=lambda: app.logout())
        self.logout_button.place(x=700, y=560)

    def open_upload_window(self):
        self.file_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a file",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
        if self.file_path:
            self.original_image = Image.open(self.file_path)
            self.original_image = self.original_image.resize((200, 200))
            self.filtered_image = None
            self.update_image()
            messagebox.showinfo("Upload Successful", "File uploaded successfully!")
        else:
            messagebox.showwarning("Upload Failed", "Please select a file to upload.")

    def show_filtered_image(self):
        if self.original_image:
            img = cv2.imread(self.file_path)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_clahe = clahe.apply(gray)
            # Apply unsharp mask filter
            blurred = cv2.GaussianBlur(gray_clahe, (0, 0), 3)
            sharp = cv2.addWeighted(gray_clahe, 0.2, blurred, 0.05, 0)
            # Apply histogram equalization
            histeq = clahe.apply(sharp)
            # Apply Gaussian blur and Sobel filter
            blurred = cv2.GaussianBlur(sharp, (3, 3), 0)
            sobel_x = cv2.Sobel(gray_clahe, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_clahe, cv2.CV_64F, 0, 1, ksize=3)
            img_sobel = cv2.addWeighted(sobel_x, 2, sobel_y, 2, 0)

            cv2.imwrite("/Users/monkey/Public/Python/Diploma_1/Main/filtered.jpg", img_sobel)
            tmp = Image.fromarray(img_sobel)
            self.filtered_image = tmp.resize((200, 200))
            self.update_image()
            # display filtered image
            filtered_image_tk = ImageTk.PhotoImage(self.filtered_image)
            self.filtered_image_label.configure(image=filtered_image_tk)
            self.filtered_image_label.image = filtered_image_tk
        else:
            messagebox.showwarning("Image Not Found", "Please download the image first.")

    def show_result_image(self):
        if self.original_image:
            img = cv2.imread('/Users/monkey/Public/Python/Diploma_1/Main/filtered.jpg')
            #img = cv2.imread(self.file_path)
            results = model(img, stream=True)
            img = cv2.imread(self.file_path)
            # creating cof for resized img
            cof_x = 400 / img.shape[1]
            cof_y = 400 / img.shape[0]
            new_size = (400, 400)
            resized_image = cv2.resize(img, new_size)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1 * cof_x - 2), int(y1 * cof_y - 2), int(x2 * cof_x + 2), int(y2 * cof_y + 2)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h))
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    current_class = classNames[cls]
                    print(current_class)

                    if conf > 0.35:
                        print("current_class")
                        my_color = (0, 0, 255)  # Red
                        cvzone.putTextRect(resized_image, f'{classNames[cls]} {conf}',
                                           (max(0, x1), max(35, y1)), scale=1, thickness=2, colorT=(255, 255, 255),
                                           offset=1)
                        cv2.rectangle(resized_image, (x1, y1), (x2, y2), my_color, 1)

            pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

            result_image_tk = ImageTk.PhotoImage(pil_image)
            self.result_image_label.configure(image=result_image_tk)
            self.result_image_label.image = result_image_tk
        else:
            messagebox.showwarning("Image Not Found", "Please upload an image first.")

    def update_image(self):
        # update original image label with current original image
        if self.original_image:
            original_image_tk = ImageTk.PhotoImage(self.original_image)
            self.original_image_label.configure(image=original_image_tk)
            self.original_image_label.image = original_image_tk

        # update filtered image label with current filtered image
        if self.filtered_image:
            filtered_image_tk = ImageTk.PhotoImage(self.filtered_image)
            self.filtered_image_label.configure(image=filtered_image_tk)
            self.filtered_image_label.image = filtered_image_tk

    def logout(self):
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    root.geometry("800x600")  # set window size
    root.title("Fracture Detector")
    app = FileUploadApp(root)

    #  start GUI loop
    root.mainloop()
