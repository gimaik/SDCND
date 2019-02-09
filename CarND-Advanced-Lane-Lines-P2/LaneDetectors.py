import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from math import log2


def eval_poly(x, fit):
	"""
	Helper function to evaluate a given polynomial function and input
	inputs:
		x: inputs to the polynomial function
		fit: polynomial function
	returns: value of the polynomial function
	"""
	degree = len(fit)-1
	return sum([(x ** (degree - i)) * fit[i] for i in range(degree + 1)])

def exp_smooth(x, smoothed_previous, gamma):
	"""
	Exponentially smooth a value
	Inputs: 
		x: value to be smoothed
		smoothed_previous: previous smoothed value
		gamma: smoothing parameter
	Returns:
		gamma * x + (1-gamma) * previous
	"""
	return gamma * x + (1-gamma) * smoothed_previous

class LaneDetector:

	"""
	This class implements lane detection via sliding window. 
	For videos, the lane lines are smoothed across the previous frames.
	Args:
		primary: if true, then primary detection only, else will implement secondary detection
		n_windows: number of sliding windows
		margin: window width = center +/- margin
		minpix: min number of pixels in the window in order to refit the curve
		smooth_gamma: smoothing parameter, [0,1]
	"""

	def __init__(self, primary=True, n_windows=10, margin=100, minpix = 50, smoother_gamma=0.20):
		self.primary = primary
		self.n_windows = n_windows
		self.margin = margin
		self.minpix = minpix
		self.window_height = 0
		self.left_lane_inds = []
		self.right_lane_inds = []
		self.previous_fit = None
		self.smoother_gamma = smoother_gamma
		self.left_curvature = None
		self.right_curvature = None

	def detect_lanes(self, img, img_only=False):
		if ((self.previous_fit != None) and (self.primary == False)):

			radius = (self.left_curvature + self.right_curvature)/2

			if radius < 1000:
				curvatures, fits, deviation, output_image = self._primary_detection(img)
			else:
				curvatures, fits, deviation, output_image = self._secondary_detection(img)
		else:
			curvatures, fits, deviation, output_image = self._primary_detection(img)

		if not self.primary:
			self.previous_fit = fits

		left_fit, right_fit = fits[0], fits[1]
		left_curvature, right_curvature = curvatures[0], curvatures[1]

		if img_only:
			return output_image
		else:
			return left_fit, left_curvature, right_fit, right_curvature, deviation, output_image

	def _initialize_lane(self, img):
		"""
		Initialize lane line locations. We need to first detect where the lane lines are.
		We compute the point in the right half of the image and then the point in the left half of the image.
		This is done by locating where the histogram reaches a maximum

		Args:
			img: image on which the lane lines are to be detected. This input image is after preprocessing
		Returns:
			A tuple (leftx_base, lefty_base) corresponding to where the lane lines start
		"""

		histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
		midpoint = np.int(histogram.shape[0]//2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint
		return leftx_base, rightx_base

	def _image_nonzero(self, img):
		"""
		Get nonzero elements in an image
		"""
		nonzero = img.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		return nonzerox, nonzeroy

	def _init_lane_pixels(self, x_low, x_high, y_low, y_high, nonzerox, nonzeroy):
		"""
		Determine the pixels that are part of a lane line within a window		
		"""
		pixels = ((nonzeroy >= y_low) & (nonzeroy < y_high) & (nonzerox >= x_low) &  (nonzerox < x_high))
		return pixels.nonzero()[0]


	def _primary_detection(self, img):
		"""
		Detect the lane using histogram and sliding windows. 
		This function is meant for single image or the first frame of a video.
		Args: 
			img: preprocessed input image
		Returns:
			left_curvature: curvature of left lane
			right_curvature: curvature of right lane
			left_fit: polynomial fitted to left lane
			right_fit: polynomial fitted to right lane
			deviation: deviation of lane center to car center
			output_img: annotated output image
		"""
		
		self.window_height = np.int(img.shape[0] // self.n_windows)
		leftx_base, rightx_base = self._initialize_lane(img)		
		leftx_current, rightx_current = leftx_base, rightx_base		
		nonzerox, nonzeroy =self._image_nonzero(img)		

		left_windows = []
		right_windows = []
		self.left_lane_inds = []
		self.right_lane_inds = []

		for window in range(self.n_windows):			
			y_low = img.shape[0] - (window + 1) * self.window_height
			y_high = img.shape[0] - window * self.window_height			
			xleft_low = leftx_current - self.margin
			xleft_high = leftx_current + self.margin
			xright_low = rightx_current - self.margin
			xright_high = rightx_current + self.margin						
			left_windows.append((xleft_low, xleft_high, y_low, y_high))
			right_windows.append((xright_low, xright_high, y_low, y_high))
		
			good_left_inds = self._init_lane_pixels(xleft_low, xleft_high, y_low, y_high, nonzerox, nonzeroy)
			good_right_inds = self._init_lane_pixels(xright_low, xright_high, y_low, y_high, nonzerox, nonzeroy)
			self.left_lane_inds.append(good_left_inds)
			self.right_lane_inds.append(good_right_inds)
			
			if len(good_left_inds) > self.minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > self.minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
		
		self.left_lane_inds = np.concatenate(self.left_lane_inds)
		self.right_lane_inds = np.concatenate(self.right_lane_inds)
	
		# Extract left and right line pixel positions
		leftx = nonzerox[self.left_lane_inds]
		lefty = nonzeroy[self.left_lane_inds] 
		rightx = nonzerox[self.right_lane_inds]
		righty = nonzeroy[self.right_lane_inds]

		# Fit a quadratic curve to the lanes
		left_fit = self._fit_polynomial(leftx, lefty) 
		right_fit = self._fit_polynomial(rightx, righty)
		self.previous_fit = (left_fit, right_fit)

		output_img= self._draw_primary_detection(img, left_windows, self.left_lane_inds, left_fit,
													right_windows, self.right_lane_inds, right_fit, 
													nonzerox, nonzeroy)

		left_center, left_curvature = self._compute_curvature(img, leftx, lefty)
		right_center, right_curvature = self._compute_curvature(img, rightx, righty)

		curvature_text = "Radius of Curvature: {:.2f} m".format(0.5*(left_curvature + right_curvature))
		cv2.putText(output_img, curvature_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)


		deviation = (img.shape[1]/2 * 3.7/700) - np.mean((left_center, right_center))
		deviation_text = "Offset: {:.2f} m".format(deviation)
		cv2.putText(output_img, deviation_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)

		self.left_curvature, self.right_curvature = left_curvature, right_curvature
		return ((left_curvature, right_curvature), (left_fit, right_fit), deviation, output_img)

	def _fit_polynomial(self, x, y, x_conv=1, y_conv=1):
		"""
		Fit a polynomial of degree 2 to a lane line
		Args:
			x: Nonzero positions along the x-axis
			y: Nonzero positions along the y-axis
		Returns:
			A polynomial fit for the lane line
		"""			
		fit = np.polyfit(y*y_conv, x*x_conv, deg=2)		
		return fit

	def _compute_curvature(self, img, img_lx, img_ly):
		"""
		Compute the radius of curvature fiven the left, right fit
		Args:
			img: image
			img_lx: nonzero along x
			img_ly: nonzero along y		
		Returns:
			curvature
		"""	
		y = img.shape[0]
		ym_per_pix = 30 / 720
		xm_per_pix = 3.7 / 700

		fit = self._fit_polynomial(img_lx, img_ly, x_conv=xm_per_pix, y_conv=ym_per_pix)
		curvature = lambda y, a, b: ((1 + (2*a*y*ym_per_pix + b)**2)**(3/2)) / np.absolute(2*a)    
		return (eval_poly(y*ym_per_pix, fit), curvature(y, fit[0], fit[1]))

	def _draw_primary_detection(self, binary_image, left_windows, left_pixels, left_fit, 
								right_windows, right_pixels, right_fit, image_1x, image_1y):
		"""
		Helper function to draw the detected lane
		"""
		output_image = np.dstack((binary_image, binary_image, binary_image))*255
		plot_y = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0], dtype=np.int32)
		left_fit_x = np.array(eval_poly(plot_y, left_fit), dtype=np.int32)
		right_fit_x = np.array(eval_poly(plot_y, right_fit), dtype=np.int32)

		output_image[image_1y[left_pixels], image_1x[left_pixels]] = [255, 0, 0]
		output_image[image_1y[right_pixels], image_1x[right_pixels]] = [0, 0, 255]
		
		for window in left_windows:
			cv2.rectangle(output_image, 
						  (window[0], window[2]),
						  (window[1], window[3]),
						  (0, 255, 0), 2)

		for window in right_windows:
			cv2.rectangle(output_image, 
						  (window[0], window[2]),
						  (window[1], window[3]),
						  (0, 255, 0), 2)

		for left_fit_point in zip(left_fit_x, plot_y):
			cv2.circle(output_image, left_fit_point, 2, (255, 255, 0))

		for right_fit_point in zip(right_fit_x, plot_y):
			cv2.circle(output_image, right_fit_point, 2, (255, 255, 0))
		return output_image


	def _secondary_detection(self, img):
		"""
		Detect the lane by making use of the prior fitted line
		This function is meant for lane detection on the subsequent frames after we used the primary detection for the first frame.
		Args: 
			img: preprocessed input image
		Returns:
			left_curvature: curvature of left lane
			right_curvature: curvature of right lane
			left_fit: polynomial fitted to left lane
			right_fit: polynomial fitted to right lane
			deviation: deviation of lane center to car center
			output_img: annotated output image
		"""


		nonzerox, nonzeroy =self._image_nonzero(img)
		left_fit, right_fit = self.previous_fit    
		self.left_lane_inds = ((nonzerox > (eval_poly(nonzeroy, left_fit) - self.margin)) & (nonzerox < (eval_poly(nonzeroy, left_fit) + self.margin)))
		self.right_lane_inds = ((nonzerox > (eval_poly(nonzeroy, right_fit) - self.margin)) & (nonzerox < (eval_poly(nonzeroy, right_fit) + self.margin)))


		if len(self.left_lane_inds) > self.minpix:
			leftx = nonzerox[self.left_lane_inds]
			lefty = nonzeroy[self.left_lane_inds]
			left_fit = self._fit_polynomial(leftx, lefty)
		else:
			left_fit = self.previous_fit[0]

		if len(self.right_lane_inds) > self.minpix:
			rightx = nonzerox[self.right_lane_inds]
			righty = nonzeroy[self.right_lane_inds]
			right_fit =self._fit_polynomial(rightx, righty)
		else:
			right_fit = self.previous_fit[0]


		left_fit = [exp_smooth(x, s_p, self.smoother_gamma) for x, s_p in zip(left_fit, self.previous_fit[0])]
		right_fit = [exp_smooth(x, s_p, self.smoother_gamma) for x, s_p in zip(right_fit, self.previous_fit[1])]
		self.previous_fit = (left_fit, right_fit)

		output_img = self._draw_secondary_detection(img, self.left_lane_inds, left_fit, self.right_lane_inds, right_fit, nonzerox, nonzeroy)
		
		left_center, left_curvature = self._compute_curvature(img, leftx, lefty)
		right_center, right_curvature = self._compute_curvature(img, rightx, righty)

		curvature_text = "Radius of Curvature: {:.2f} m".format(0.5*(left_curvature + right_curvature))
		cv2.putText(output_img, curvature_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)

		deviation = (img.shape[1]/2 * 3.7/700) - np.mean((left_center, right_center))
		deviation_text = "Offset: {:.2f} m".format(deviation)
		cv2.putText(output_img, deviation_text, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)

		self.left_curvature, self.right_curvature = left_curvature, right_curvature
		return ((left_curvature, right_curvature), (left_fit, right_fit), deviation, output_img)


	def _draw_secondary_detection(self, img, left_lane, left_fit, right_lane, right_fit, image_1x, image_1y):
		"""
		Helper function to draw the detected lane
		"""
		output_img = np.dstack((img, img, img))*255
		window_img = np.zeros_like(output_img)
		output_img[image_1y[self.left_lane_inds], image_1x[self.left_lane_inds]] = [255, 0, 0]
		output_img[image_1y[self.right_lane_inds], image_1x[self.right_lane_inds]] = [0, 0, 255]

		plot_y = np.linspace(0, img.shape[0]-1, img.shape[0])

		left_fitx = np.array(eval_poly(plot_y, left_fit), dtype=np.int32)
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, plot_y]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin, plot_y])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))

		right_fitx = np.array(eval_poly(plot_y, right_fit), dtype=np.int32)
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, plot_y]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin, plot_y])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		for left_fit_point in zip(left_fitx, plot_y):
			lx, ly = left_fit_point
			lx, ly = int(lx), int(ly)
			cv2.circle(output_img, (lx, ly), 1, (255, 255, 0))

		for right_fit_point in zip(right_fitx, plot_y):
			rx, ry = left_fit_point
			rx, ry = int(rx), int(ry)
			cv2.circle(output_img, (rx, ry), 1, (255, 255, 0))

		output_img = cv2.addWeighted(output_img, 1, window_img, 0.3, 0)

		return output_img