import numpy as np
import lne_tflite as lt
import os, sys
import cv2

class Network:
	def __init__(self, model_path, label_path):
		self.interpreter = lt.lite.Interpreter(model_path = model_path)
		self.interpreter.allocate_tensors()
		self.input_detail = self.interpreter.get_input_details()
		self.output_detail = self.interpreter.get_output_details()

		with open(label_path) as f:
			l_lines = f.readlines()
			self.labels = [ line for line in l_lines]

		input_shape = self.input_detail[0]['shape']
		self.width = input_shape[2]
		self.height = input_shape[1]


	def resize_img(self, img):
		fitted_img = cv2.resize(img, (self.width, self.height))
		fitted_img = cv2.cvtColor(fitted_img, cv2.COLOR_BGR2RGB)
		fitted_img = np.expand_dims(fitted_img, axis = 0)
		return fitted_img

	def crop_image(self, img):
		(y,x,channel) = img.shape
		x_prime = y
		img = img[0:y, int((x-x_prime)/2):int((x+x_prime)/2)]
		return img

	def inference(self, img):
		self.interpreter.set_tensor(self.input_detail[0]['index'], img)
		self.interpreter.invoke()
		output_lne = self.interpreter.get_tensor(self.output_detail[0]['index'])
		lne_answer_argmax = np.argmax(output_lne)
		lne_answer_argsort = np.argsort(output_lne)
		lne_answer_argsort = np.reshape(lne_answer_argsort, -1)
		lne_output = np.reshape(output_lne, -1)
		lne_answer = lne_answer_argsort[-5:]
		return lne_answer, lne_output

	def post_draw(self, orig_image, answer):
		blue = (255, 0, 0)
		thickness = 3
		center_x = orig_image.shape[1]
		center_y = orig_image.shape[0]
		location = ((center_x // 2) - 250, (center_y // 2) + 200)
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontScale= 2
		orig_image = cv2.flip(orig_image, 1)
		cv2.putText(orig_image, self.labels[answer].strip(), location, font, fontScale, blue, thickness)
		bmp = cv2.imencode('.bmp', orig_image)[1].tobytes()
		return bmp
