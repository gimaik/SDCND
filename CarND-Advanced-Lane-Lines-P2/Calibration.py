import cv2
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Calibration:
	def __init__(self, nx, ny, image_paths):
		self.nx = nx
		self.ny = ny
		self.path = image_paths
		self.mtx = None
		self.dist = None
		self.rvecs = None
		self.tvecs = None
		self.__calibrator()

	def __calibrator(self):		
		objp = np.zeros((self.nx * self.ny,3), dtype=np.float32)
		objp[:,:2] = np.mgrid[0 : self.nx, 0 : self.ny].T.reshape(-1,2)
		
		objpoints = [] 
		imgpoints = [] 

		for idx, fname in enumerate(self.path):    
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
	
			if ret:
				objpoints.append(objp)
				imgpoints.append(corners)
				cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)               

		ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)

		return None


	def undistort(self, img):
		undistort_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
		return undistort_img
    
	def plot_images(self, fname):
		img = mpimg.imread(fname)
		undistort_img = self.undistort(img)
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
		f.tight_layout()
		ax1.imshow(img)
		ax1.set_title('Original Image', fontsize=15)
		ax2.imshow(undistort_img)
		ax2.set_title('Undistorted Image', fontsize=15)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		return undistort_img