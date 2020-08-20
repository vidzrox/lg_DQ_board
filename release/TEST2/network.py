import numpy as np
import lne_tflite as lt
import os, sys
import cv2

class Network:
	def __init__(self, model_path):
		self.interpreter = lt.lite.Interpreter(model_path = model_path)
		self.interpreter.allocate_tensors()
		self.input_detail = self.interpreter.get_input_details()
		self.output_detail = self.interpreter.get_output_details()

		input_shape = self.input_detail[0]['shape']
		self.width = input_shape[2]
		self.height = input_shape[1]

	def open_image(self, input_path):
		img = cv2.imread(input_path)
		return img

	def crop_image(self, img):
		(y,x,channel) = img.shape
		x_prime = y
		img = img[0:y, int((x-x_prime)/2):int((x+x_prime)/2)]
		return img

	def resize_img(self, img):
		fitted_img = cv2.resize(img, (self.width, self.height))
		fitted_img = cv2.cvtColor(fitted_img, cv2.COLOR_BGR2RGB)	
		fitted_img = np.expand_dims(fitted_img, axis = 0)
		return fitted_img
	
	def resize_img_rgb(self, img):
		fitted_img = cv2.resize(img, (self.width, self.height))
		fitted_img = np.expand_dims(fitted_img, axis = 0)
		return fitted_img

	def inference(self, img):
		self.interpreter.set_tensor(self.input_detail[0]['index'], img)
		self.interpreter.invoke()
		output_lne = self.interpreter.get_tensor(self.output_detail[0]['index'])
		output_image = np.round((np.tanh(output_lne)+1)*127.5)
		return output_image

	def post_draw(self, orig_image, answer):
		lne_answer = np.squeeze(answer, axis = 0)
		orig_image = cv2.resize(orig_image, (self.width, self.height))
		lne_answer = cv2.cvtColor(lne_answer, cv2.COLOR_RGB2BGR)
		lne_answer = cv2.flip(lne_answer, 1)
		orig_image = cv2.flip(orig_image, 1)
		bmp_lne = cv2.imencode('.bmp', lne_answer)[1].tobytes()
		bmp_org = cv2.imencode('.bmp', orig_image)[1].tobytes()
		cv2.imwrite("image_lne.jpg", lne_answer)
		cv2.imwrite("image_org.jpg", orig_image)
		return bmp_org, bmp_lne

	def post_draw_rgb(self, orig_image, answer):
		lne_answer = np.squeeze(answer, axis = 0)
		orig_image = cv2.resize(orig_image, (self.width, self.height))
		lne_answer = cv2.flip(lne_answer, 1)
		orig_image = cv2.flip(orig_image, 1)
		bmp_lne = cv2.imencode('.bmp', lne_answer)[1].tobytes()
		bmp_org = cv2.imencode('.bmp', orig_image)[1].tobytes()
		cv2.imwrite("image_lne.jpg", lne_answer)
		cv2.imwrite("image_org.jpg", orig_image)
		return bmp_org, bmp_lne

