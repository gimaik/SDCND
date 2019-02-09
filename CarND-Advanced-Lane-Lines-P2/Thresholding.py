import numpy as np
import cv2

def preprocess(img):
	"""
	Preprocessing pipeline: CLAHE -> Gaussian smoothing -> HLS channel extraction
	Args:
		img: Input image in BGR colorspace
	Return: S channel of the input image after preprocessing
	"""
	img_work = CLAHE(img, clip_limit=2, tile_grid_size=(4,4))
	img_work = cv2.GaussianBlur(img_work, (3, 3), 0)
	img_work = cv2.cvtColor(img_work, cv2.COLOR_RGB2HLS)
	img_work = img_work[:,:,2]
	return img_work

def CLAHE(img, clip_limit, tile_grid_size):
	"""
	Apply Contrast Limited Adaptive Histogram Equalization with OpenCV
	CLAHE is applied to each channels of the image and the result is returned as an RGB image
	Args:
		img: Input image in BGR colorspace
		clip_limit: pass to cv2.createCLAHE
		tile_grid_size: pass to cv2.createCLAHE
	Return: Input image with CLAHE applied in RGB
	"""
	b, g, r = cv2.split(img)
	img_clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
	clahe_r = img_clahe.apply(r)
	clahe_g = img_clahe.apply(g)
	clahe_b = img_clahe.apply(b)
	img_ret = cv2.merge((clahe_r, clahe_g, clahe_b))
	return img_ret

def abs_sobel(img, kernel=3):
	"""
	Computing the gradient of the input image in the x and y direction
	Args:
	img: Input image after preprocessing (S channel only)
	Return: The scaled gradients of the image in x and y directions
	"""

	sobel_x = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel))
	sobel_y = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel))
	sobel_x = np.uint8(255*sobel_x / np.max(sobel_x))
	sobel_y = np.uint8(255*sobel_y / np.max(sobel_y))
	return sobel_x, sobel_y

def grad_magnitude(sobel_x, sobel_y):
	"""
	Compute the magnitude of the gradient
	Args:
	sobel_x: gradient in x
	sobel_y: gradient in y
	Return: Magnitude of the gradient of an image
	"""	
	magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
	magnitude = np.uint8(magnitude / (np.max(magnitude) / 255))	
	return magnitude

def grad_direction(sobel_x, sobel_y):	
	"""
	Compute the direction of the gradient
	Args:
	sobel_x: gradient in x
	sobel_y: gradient in y
	Return: Direction of the gradient of an image
	"""	
	direction = np.arctan2(sobel_y, sobel_x)	
	return direction

def color_threshold(img, type='LUV'):
	"""
	Convert a BGR image into Luv or Lab and extract the relevant channels
	Args:
	img: input BGR image
	Return: the relevant channels of the image
	"""	

	if type=='LUV':
		img_work = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
		img_work = img_work[:,:,0]
	else:
		img_work = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
		img_work = img_work[:,:,2]
	return img_work



def threshold(img, lower_bound, upper_bound):
	"""
	Helper function to construct binary image
	Args:
	img: image / magnitude / direction
	lower_bound: lower_bound of the threshold
	upper_bound: upper_bound of the threshold
	Return: Binary image after implementing the thresholding
	"""

	binary = np.zeros_like(img)
	binary [(img > lower_bound) & (img < upper_bound)] = 1 
	return binary

def combined_thresholds(img):
	"""
	Combining all the different thresholding criteria
	"""
	luv = color_threshold(img, "LUV")
	lab = color_threshold(img, "LAB")
	img = preprocess(img)
	grad_x, grad_y = abs_sobel(img, kernel=3)	
	grad_mag = grad_magnitude(grad_x, grad_y)
	grad_dir = grad_direction(grad_x, grad_y)
	
	luv_binary = threshold(luv, 200, 255)
	lab_binary = threshold(lab, 150, 200)
	magnitude_binary = threshold(grad_mag, 40, 100)
	direction_binary = threshold(grad_dir, 0.7, 1.30)
	gradx_binary  = threshold(grad_x, 40, 100)
	grady_binary = threshold(grad_y, 40, 100)
	binary = np.zeros_like(grad_x)	

	binary[(gradx_binary==1) | (luv_binary==1) | (lab_binary==1) | (grady_binary==1)]=1

	return binary