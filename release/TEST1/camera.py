import cv2
import numpy as np

class VideoCamera(object):
	def __init__(self, camID):
		# Using OpenCV to capture from device 0. If you have trouble capturing
		# from a webcam, comment the line below out and use a video file
		# instead.
		self.video = cv2.VideoCapture(camID)
		while not self.video.isOpened():
			self.video = cv2.VideoCapture(camID)
			cv2.waitKey(100)
			print("Wait for camera preparation ....")

		self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
		self.video.set( cv2.CAP_PROP_FRAME_HEIGHT, 480)
		# If you decide to use video.mp4, you must have this file in the folder
		# as the main.py.
		# self.video = cv2.VideoCapture('video.mp4')

	def __del__(self):
		print("destory")
		self.video.release()

	#DQ1 camera data format is planar, so convert planar to packed
	def dq1_camera_convert(self, frame):
		return np.transpose(frame.reshape(3, 480, 640), (1, 2, 0))

	def get_frame(self):
		success, image = self.video.read()
		image = self.dq1_camera_convert(image)
		return image
