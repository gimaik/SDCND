import cv2
import numpy as np 
import pandas as pd 


class PerspectiveTransform:

	"""
	Applying a perspective transform
	Args:
	source_pts: Source points for perspective transform, as defined on the original image
	destination_pts: Destination points for perspective transform, as defined on the output image
	"""


	def __init__(self, source_pts, destination_pts):
		self.source_pts = source_pts
		self.destination_pts = destination_pts
		self.transformation_matrix = cv2.getPerspectiveTransform(source_pts, destination_pts)
		self.inverse_transformation_matrix = cv2.getPerspectiveTransform(destination_pts, source_pts)



	def warp(self, image):
		"""
		Transform image perspective using OpenCV functions
		Args: Input image
		Return: Input image with the perspective transformed
		"""
		return cv2.warpPerspective(image, self.transformation_matrix, (image.shape[0], image.shape[1]), flags=cv2.INTER_CUBIC)


	def unwarp(self, image):
		"""
		Inverse transform image perspective using OpenCV functions
		Args: Input image
		Return: Input image with the inverse perspective transform applied
		"""	
		return cv2.warpPerspective(image, self.inverse_transformation_matrix, (image.shape[0], image.shape[1]), flags=cv2.INTER_CUBIC)