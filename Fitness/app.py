import os
import tkinter as tk

from PIL import Image
import PIL.ImageTk
import cv2
import model


import camera


class App:

    def __init__(self):
        self.window = tk.Tk()  # this is going to create the window
        self.window.title = "Godson Bicep Rep Counter"

        self.counters = [1, 1]  # counter for sample images, trained on pictures of extended arm and contracted arm
        self.rep_counter = 0

        # In order to count reps we need to find the transition
        self.extended = False
        self.contracted = False
        self.last_prediction = 0

        self.model = model.Model()

        self.counting_enabled = False

        self.camera = camera.Camera()

        self.init_gui()  # function for initialization of the gui

        self.delay = 15  # delay for gui
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    # gui should have a part where you can see the camera, button that can
    # add new examples for contracted and extended arms for resetting the score and training the model

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.pack()

        self.btn_toggleauto = tk.Button(self.window, text="Toggle Counting", width=50, command=self.counting_toggle)
        self.btn_toggleauto.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_one = tk.Button(self.window, text="Extended", width=50, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(anchor=tk.CENTER, expand=True)

        self.btn_class_two = tk.Button(self.window, text="Contracted", width=50, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(anchor=tk.CENTER, expand=True)

        self.btn_train = tk.Button(self.window, text="Train Model", width=50,
                                   command=lambda: self.model.train_model(self.counters))
        self.btn_train.pack(anchor=tk.CENTER, expand=True)

        self.btn_reset = tk.Button(self.window, text="Reset", width=50, command=self.reset)
        self.btn_reset.pack(anchor=tk.CENTER, expand=True)

        self.counter_label = tk.Label(self.window, text=f"{self.rep_counter}")
        self.counter_label.config(font={"Arial", 24})
        self.counter_label.pack(anchor=tk.CENTER, expand=True)

    def update(self):
        if self.counting_enabled:
            self.predict()

        if self.extended and self.contracted:
            self.extended, self.contracted = False, False
            self.rep_counter += 1

        self.counter_label.config(text=f"{self.rep_counter}")

        ret, frame = self.camera.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    #  predict function sets the value so the update function knows when to count up
    def predict(self):  # this is going cause the model to predict the current image
        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)

        #  checking if the prediction we have now is not the same as the last prediction we made, something changed
        if prediction != self.last_prediction:
            if prediction == 1:
                self.extended = True
                self.last_prediction = 1
            if prediction == 2:
                self.contracted = True
                self.last_prediction = 2

    def counting_toggle(self):
        self.counting_enabled = not self.counting_enabled
        pass

    def save_for_class(self, class_num):  # this take the image we already have and saves it as a training example
        ret, frame = self.camera.get_frame()
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        #  write the image into the file path
        cv2.imwrite(f"{class_num}/frame{self.counters[class_num - 1]}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

        # saving it with open cv, loading it with pillow, resizing it then saving with pillow again
        img = PIL.Image.open(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")

        self.counters[class_num - 1] += 1

    def reset(self):
        self.rep_counter = 0
        pass
