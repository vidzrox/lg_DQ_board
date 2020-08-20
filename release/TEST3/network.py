import numpy as np
import lne_tflite as lt
import os, sys
import cv2
from result import detect_faces, draw_text, draw_bounding_box, apply_offsets

class Network:
	def __init__(self, model_path):
		self.interpreter = lt.lite.Interpreter(model_path = model_path)
		self.interpreter.allocate_tensors()
		self.input_detail = self.interpreter.get_input_details()
		self.output_detail = self.interpreter.get_output_details()
		self.labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
		input_shape = self.input_detail[0]['shape']
		self.width = input_shape[2]
		self.height = input_shape[1]


	def resize_img(self, img):
		fitted_img = cv2.resize(img, (self.width, self.height))
		fitted_img = np.expand_dims(fitted_img, axis = 0)
		fitted_img = np.expand_dims(fitted_img, axis = -1)
		return fitted_img

	def inference(self, img):
		self.interpreter.set_tensor(self.input_detail[0]['index'], img)
		self.interpreter.invoke()
		output_lne = self.interpreter.get_tensor(self.output_detail[0]['index'])
		lne_answer = np.argmax(output_lne)
		prob = output_lne.flatten()[lne_answer]*100
		return lne_answer, prob

	def post_draw(self, orig_image):
		orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
		#orig_image = cv2.flip(orig_image, 1)
		bmp = cv2.imencode('.bmp', orig_image)[1].tobytes()
		return bmp

	def face_detect(self, img):
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		face_detection = cv2.CascadeClassifier("/home/ubuntu/app_example/models/haarcascade_frontalface_default.xml")
		face = detect_faces(face_detection, gray_img)
		return face, gray_img
