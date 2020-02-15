import cv2
import sys
import numpy as np
import os
import argparse
import time
from datetime import datetime
from sklearn.preprocessing import normalize
import open3d

ply_header = ('''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
)


def write_ply(output_file, verts, colors):
	"""Export ``PointCloud`` to PLY file for viewing in MeshLab."""
	verts = verts.reshape(-1, 3)
	colors = colors.reshape(-1, 3)
	points = np.hstack([verts, colors])
	with open(output_file, 'w') as outfile:
		outfile.write(ply_header.format(vertex_count=len(verts)))
		outfile.write("\n")
		np.savetxt(outfile, points, '%f %f %f %d %d %d')


class ZED(object):
	def __init__(self):
		self.fromFile = False
		self.cameras_opened =  False

		# Arrays to store object points and image points from all the images.
		self.frameL = []
		self.frameR = []
		self.frameStereoL = []
		self.frameStereoR = []
		# create arrays to store object points and image points from all the images.
		self.objpoints = [] # 3d point in real world space
		self.objpointsStereo = [] # 3d point in real world space
		self.imgpointsR = [] # 2d points in image plane.
		self.imgpointsL = [] # 2d points in image plane.
		self.imgpointsStereoR = [] # 2d points in image plane.
		self.imgpointsStereoL = [] # 2d points in image plane.
		self.distList = [] #  distortion patameters
		self.mtxL = []
		self.distL = []
		self.rvecsL = [] 
		self.tvecsL = []
		self.mtxR = []
		self.distR = []
		self.rvecsR = [] 
		self.tvecsR = []
		self.chessboard_pattern_detections = 0
		self.patternX = 5
		self.patternY= 9
		self.square_size_in_mm = 18
		self.calibration_capture_framerate = 2
		self.disparity_processing_framerate = 25
		self.keep_processing = True
		self.do_calibration = False
		self.windowNameL = "LEFT Camera Input" # window name
		self.windowNameR = "RIGHT Camera Input" # window name
		self.imageShape = []
		self.rms_stereo = []
		self.camera_matrix_l = []
		self.dist_coeffs_l = []
		self.camera_matrix_r = []
		self.dist_coeffs_r = []
		self.R = []
		self.T = []
		self.E = []
		self.F = []
		self.RL = []
		self.RR = []
		self.PL = []
		self.PR = []
		self.Q  = []
		self.mapL1 = []
		self.mapL2 = []
		self.mapR1 = []
		self.mapR2 = []
		self.maxImageCount = 40
		self.newcameramtxL = []
		self.newcameramtxR = []
		self.CalibrationFile = False
		self.Xpoint = None
		self.YPoint = None
		self.VO = True
		self.VOImageCount = 100
		self.isZed = False
		self.sequence = 0
		self.leftImagePath = None
		self.rightImagePath = None
		self.calibFileName = None
		self.Proj1 = None
		self.Proj2 = None
		self.groundTruthTraj = []

	def ReadCalibrationFile(self):
		if self.isZed is True:
			outputfile = "input/ZEDCalibration.npz"
			calibration = np.load(outputfile, allow_pickle=False)
			self.imageShape = tuple(calibration["imageSize"])
			self.mapL1 = calibration["leftMapX"]
			self.mapL2 = calibration["leftMapY"]
			leftROI = tuple(calibration["leftROI"])
			self.mapR1 = calibration["rightMapX"]
			self.mapR2 = calibration["rightMapY"]
			rightROI = tuple(calibration["rightROI"])
			self.Q = calibration["Qvalue"]
		else:

			calibFile = open(self.calibFileName, 'r').readlines()
			P1Vals = calibFile[0].split()
			self.Proj1 = np.zeros((3,4))
			for row in range(3):
				for column in range(4):
					self.Proj1[row, column] = float(P1Vals[row*4 + column + 1])

			P2Vals = calibFile[1].split()
			self.Proj2 = np.zeros((3,4))
			for row in range(3):
				for column in range(4):
					self.Proj2[row, column] = float(P2Vals[row*4 + column + 1])
	def AcquireFrames(self):
		
		self.camZED = cv2.VideoCapture()
		if not(self.camZED.open(0)):
			print("Cannot open connected ZED stereo camera as camera #: ", 0)
			exit()

		_, frame = self.camZED.read()
		height,width, channels = frame.shape
		cameras_opened = True
		# grab single frame from camera (read = grab/retrieve)

		frameL= frame[:,0:int(width/2),:]
		frameR = frame[:,int(width/2):width,:]
		return frameL, frameR

	def DisplayFrames(self,imgL,imgR):


		# create window by name (as resizable)

		cv2.namedWindow(self.windowNameL, cv2.WINDOW_NORMAL)
		cv2.namedWindow(self.windowNameR, cv2.WINDOW_NORMAL)

		# set sizes and set windows

		height, width, channels = imgL.shape
		height = int(height/2)
		width = int(width /2)
		cv2.resizeWindow(self.windowNameL, width, height)
		#height, width, channels = imgR.shape
		cv2.resizeWindow(self.windowNameR, width, height)

	def FindImagePoints(self):
		# STAGE 2: perform intrinsic calibration (removal of image distortion in each image)

		termination_criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# set up a set of real-world "object points" for the chessboard pattern

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

		objp = np.zeros((self.patternX*self.patternY,3), np.float32)
		objp[:,:2] = np.mgrid[0:self.patternX,0:self.patternY].T.reshape(-1,2)
		objp = objp * self.square_size_in_mm

		

		# count number of chessboard detection (across both images)
		self.chessboard_pattern_detections = 0

		print()
		print("--> processing input images")
		print()
		while (not(self.do_calibration)):
			# get frames from camera
			if (self.fromFile == False):
				frameL, frameR = self.AcquireFrames()
			else:
				frameL, frameR = self.frameL[self.chessboard_pattern_detections], self.frameR[self.chessboard_pattern_detections]

			# convert to grayscale

			grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
			grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
			self.imageShape = grayL.shape
			# Find the chess board corners in the image
			# (change flags to perhaps improve detection ?)

			retL, cornersL = cv2.findChessboardCorners(grayL, (self.patternX,self.patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
			retR, cornersR = cv2.findChessboardCorners(grayR, (self.patternX,self.patternY),None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

			# If found, add object points, image points (after refining them)

			if ((retR == True) and (retL == True)):
				self.chessboard_pattern_detections += 1
				now = datetime.now()
				dt_string = now.strftime("%d%m%Y_%H%M%S")

				outputfile = "calibImage/ImageL{:02d}.jpg".format(self.chessboard_pattern_detections)
				cv2.imwrite(outputfile, frameL)
				outputfile = "calibImage/ImageR{:02d}.jpg".format(self.chessboard_pattern_detections)
				cv2.imwrite(outputfile, frameR)				

				# add object points to global list

				self.objpoints.append(objp)

				# refine corner locations to sub-pixel accuracy and then

				corners_sp_L = cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),termination_criteria_subpix)
				self.imgpointsStereoL.append(corners_sp_L)
				corners_sp_R = cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),termination_criteria_subpix)
				self.imgpointsStereoR.append(corners_sp_R)
				self.imgpointsL.append(corners_sp_L)
				self.imgpointsR.append(corners_sp_R)
				# Draw and display the corners

				drawboardL = cv2.drawChessboardCorners(frameL, (self.patternX,self.patternY), corners_sp_L,retL)
				drawboardR = cv2.drawChessboardCorners(frameR, (self.patternX,self.patternY), corners_sp_R,retR)

				text = 'detected: ' + str(self.chessboard_pattern_detections)
				cv2.putText(drawboardL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)

				outputfile = "calibImage/ImagewithCornersL{:02d}.jpg".format(self.chessboard_pattern_detections)
				cv2.imwrite(outputfile, drawboardL)		
				outputfile = "calibImage/ImagewithCornersR{:02d}.jpg".format(self.chessboard_pattern_detections)
				cv2.imwrite(outputfile, drawboardR)		

				cv2.imshow(self.windowNameL,drawboardL)
				cv2.imshow(self.windowNameR,drawboardR)
			else:
				text = 'detected: ' + str(self.chessboard_pattern_detections)
				cv2.putText(frameL, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 8)

				cv2.imshow(self.windowNameL,frameL)
				cv2.imshow(self.windowNameR,frameR)
			
			key = cv2.waitKey(int(4000/self.calibration_capture_framerate)) & 0xFF # wait 500ms between frames - i.e. 2 fps
			if (self.chessboard_pattern_detections >= self.maxImageCount or key == ord(' ')):
				self.keep_processing = False
				self.do_calibration = True
				return grayL, grayR		
	def Calibrate(self,grayL, grayR):
		if (self.chessboard_pattern_detections >= self.maxImageCount ): # i.e. if we did not load a calibration

			print("START - intrinsic calibration ...")
			h,  w = self.imageShape

			ret, self.mtxL, self.distL, self.rvecsL, self.tvecsL = cv2.calibrateCamera(self.objpoints, self.imgpointsL, (w,h),None,None)
			ret, self.mtxR, self.distR, self.rvecsR, self.tvecsR = cv2.calibrateCamera(self.objpoints, self.imgpointsR, (w,h),None,None)

			print("FINISHED - intrinsic calibration")

			# perform undistortion of the images

			self.keep_processing = True
			print()
			print("self.mtxL")
			print(self.mtxL)
			print("self.distL")
			print(self.distL)
			print("self.mtxR")
			print(self.mtxR)
			print("self.distR")
			print(self.distR)
		
			print()
			print("-> dislaying undistortion")
			print("press space to continue to next stage ...")
			print()

			while (self.keep_processing):

				# get frames from camera

				frameL, frameR = self.AcquireFrames()
				
				h,  w = self.imageShape
				
				self.newcameramtxL, roiL = cv2.getOptimalNewCameraMatrix(self.mtxL, self.distL, (w,h), 1, None)
				self.newcameramtxR, roiR = cv2.getOptimalNewCameraMatrix(self.mtxR, self.distR, (w,h), 1, None)
				print("self.newcameramtxL")
				print(self.newcameramtxL)
				print("self.newcameramtxR")
				print(self.newcameramtxR)
				print("roiL")
				print(roiL)
				print("roiR")
				print(roiR)

				undistortedL = cv2.undistort(frameL, self.mtxL, self.distL, None, self.newcameramtxL)
				undistortedR = cv2.undistort(frameR, self.mtxR, self.distR, None, self.newcameramtxR)

				# display image

				cv2.imshow(self.windowNameL,undistortedL)
				cv2.imshow(self.windowNameR,undistortedR)

				# start the event loop - essential

				key = cv2.waitKey(int(1000/self.disparity_processing_framerate)) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

				if (key == ord(' ')):
					self.keep_processing = False
				elif (key == ord('x')):
					exit()
	def CalculateReprojectionError(self):
		if (self.chessboard_pattern_detections >=  self.maxImageCount): # i.e. if we did not load a calibration

			tot_errorL = 0
			for i in range(len(self.objpoints)):
				imgpointsL2, _ = cv2.projectPoints(self.objpoints[i], self.rvecsL[i], self.tvecsL[i], self.mtxL, self.distL)
				errorL = cv2.norm(self.imgpointsL[i],imgpointsL2, cv2.NORM_L2)/len(imgpointsL2)
				tot_errorL += errorL

			print("LEFT: Re-projection error: ", tot_errorL/len(self.objpoints))

			tot_errorR = 0
			for i in range(len(self.objpoints)):
				imgpointsR2, _ = cv2.projectPoints(self.objpoints[i], self.rvecsR[i], self.tvecsR[i], self.mtxR, self.distR)
				errorR = cv2.norm(self.imgpointsR[i],imgpointsR2, cv2.NORM_L2)/len(imgpointsR2)
				tot_errorR += errorR

			print("RIGHT: Re-projection error: ", tot_errorR/len(self.objpoints))

	def ExtrinsicCalibration(self):
		termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

		if (self.chessboard_pattern_detections > 0): # i.e. if we did not load a calibration
			print()
			print("START - extrinsic calibration ...")
			h,  w = self.imageShape
			(self.rms_stereo, self.camera_matrix_l, self.dist_coeffs_l, self.camera_matrix_r, self.dist_coeffs_r, self.R, self.T, self.E, self.F) = \
			cv2.stereoCalibrate(self.objpoints, self.imgpointsL, self.imgpointsR, self.mtxL, self.distL, self.mtxR, self.distR,  (w,h), criteria=termination_criteria_extrinsics, flags=0)
			print("camera_matrix_l")
			print (self.camera_matrix_l)
			print("mtxL")
			print(self.mtxL)
			print("camera_matrix_r")
			print (self.camera_matrix_r)
			print("mtxR")
			print(self.mtxR)

			print("self.dist_coeffs_l")
			print(self.dist_coeffs_l)
			print("self.distL")
			print(self.distL)
			print("self.dist_coeffs_r")
			print(self.dist_coeffs_r)
			print("self.distR")
			print(self.distR)
			
			print("START - extrinsic calibration ...")

			print("STEREO: RMS left to  right re-projection error: ", self.rms_stereo)

	def RectifyImages(self):
		# STAGE 4: rectification of images (make scan lines align left <-> right

		# N.B.  "alpha=0 means that the rectified images are zoomed and shifted so that
		# only valid pixels are visible (no black areas after rectification). alpha=1 means
		# that the rectified image is decimated and shifted so that all the pixels from the original images
		# from the cameras are retained in the rectified images (no source image pixels are lost)." - ?
		keep_processing = True
		if (self.chessboard_pattern_detections > 0): # i.e. if we did not load a calibration
			self.RL, self.RR, self.PL, self.PR, self.Q, leftROI, rightROI = \
			cv2.stereoRectify(
				self.camera_matrix_l, 
				self.dist_coeffs_l, 
				self.camera_matrix_r, 
				self.dist_coeffs_r,  
				(1280,720), 
				self.R, 
				self.T, 
				flags=cv2.CALIB_ZERO_DISPARITY,
				alpha=-1,
				newImageSize=(0,0)
				)
		print(self.PL)
		print(self.PR)
		print(self.Q)
		#imgpointsL = cv2.undistortPoints(self.imgpointsL, self.newcameramtxL, self.distL, P=self.camera_matrix_l) 
		#imgpointsR = cv2.undistortPoints(self.imgpointsR, self.newcameramtxR, self.distR, P=self.camera_matrix_r) 
		# compute the pixel mappings to the rectified versions of the images

		if (self.chessboard_pattern_detections > 0): # i.e. if we did not load a calibration
			self.mapL1, self.mapL2 = cv2.initUndistortRectifyMap(self.camera_matrix_l, self.dist_coeffs_l, self.RL, self.PL, (1280,720), cv2.CV_32FC1)
			#self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(self.camera_matrix_r, self.dist_coeffs_r, self.RR, self.PR, self.imageShape, cv2.CV_32FC1)
			self.mapR1, self.mapR2 = cv2.initUndistortRectifyMap(self.camera_matrix_r, self.dist_coeffs_r, self.RR, self.PR, (1280,720), cv2.CV_32FC1)
			outputfile = "input/ZEDCalibration.npz"
			np.savez_compressed(outputfile, imageSize = self.imageShape,
				leftMapX=self.mapL1, leftMapY=self.mapL2, leftROI=leftROI,
				rightMapX=self.mapR1, rightMapY=self.mapR2, rightROI=rightROI, Qvalue=self.Q)
			print()
			print("-> displaying rectification")
			print("press space to continue to next stage ...")

			

		while (keep_processing):

			# get frames from camera

			frameL, frameR = self.AcquireFrames()
			grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
			grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)

			# undistort and rectify based on the mappings (could improve interpolation and image border settings here)

			undistorted_rectifiedL = cv2.remap(grayL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
			undistorted_rectifiedR = cv2.remap(grayR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

			# display image

			cv2.imshow(self.windowNameL,undistorted_rectifiedL)
			cv2.imshow(self.windowNameR,undistorted_rectifiedR)

			# start the event loop - essential

			key = cv2.waitKey(int(1000/self.disparity_processing_framerate)) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

			# It can also be set to detect specific key strokes by recording which key is pressed

			# e.g. if user presses "x" then exit

			if (key == ord(' ')):
				keep_processing = False
			elif (key == ord('x')):
				exit()
	def CalculateStereoDepth(self):
		#####################################################################

		# STAGE 5: calculate stereo depth information

		# uses a modified H. Hirschmuller algorithm [HH08] that differs (see opencv manual)

		# parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

		# FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21)

		# From help(cv2): StereoBM_create(...)
		#        StereoBM_create([, numDisparities[, blockSize]]) -> retval
		#
		#    StereoSGBM_create(...)
		#        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
		# disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval
		if self.CalibrationFile is True:
			self.ReadCalibrationFile()
			self.windowNameL = "LEFT Camera Input" # window name
			self.windowNameR = "RIGHT Camera Input" # window name
			height, width = self.imageShape
			cv2.namedWindow(self.windowNameL, cv2.WINDOW_NORMAL)
			cv2.namedWindow(self.windowNameR, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.windowNameL, int(width/2), int(height/2))
			cv2.resizeWindow(self.windowNameR, int(width/2), int(height/2))

		#best output with the following link
		#http://timosam.com/python_opencv_depthimage
		# SGBM Parameters -----------------
		window_size = 15                     # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
		
		left_matcher = cv2.StereoSGBM_create(
			minDisparity=0,
			numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
			blockSize=5,
			P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
			P2=32 * 3 * window_size ** 2,
			disp12MaxDiff=1,
			uniquenessRatio=15,
			speckleWindowSize=0,
			speckleRange=2,
			preFilterCap=63,
			mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
		)
		right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
		# FILTER Parameters
		lmbda = 80000
		sigma = 1.2
		visual_multiplier = 1.0
		
		wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
		wls_filter.setLambda(lmbda)
		wls_filter.setSigmaColor(sigma)

		# set up disparity window to be correct size

		windowNameD = "SGBM Stereo Disparity - Output" # window name
		cv2.namedWindow(windowNameD, cv2.WINDOW_NORMAL)
		height, width = self.imageShape
		cv2.resizeWindow(windowNameD, int(width/2), int(height/2))
		cv2.setMouseCallback(windowNameD,self.draw_circle)
		
		print()
		print("-> display disparity image")
		print("press x to exit")
		print("press c for false colour mapped disparity")

		keep_processing = True
		apply_colourmap = False

		while (keep_processing):

			# get frames from camera

			frameL, frameR = self.AcquireFrames()

			# remember to convert to grayscale (as the disparity matching works on grayscale)

			grayL = cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
			grayR = cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)

			
			# undistort and rectify based on the mappings (could improve interpolation and image border settings here)
			# N.B. mapping works independant of number of image channels

			undistorted_rectifiedL = cv2.remap(grayL, self.mapL1, self.mapL2, cv2.INTER_LINEAR)
			undistorted_rectifiedR = cv2.remap(grayR, self.mapR1, self.mapR2, cv2.INTER_LINEAR)

			# compute disparity image from undistorted and rectified versions
			# (which for reasons best known to the OpenCV developers is returned scaled by 16)

			#print('computing disparity...')
			displ = left_matcher.compute(undistorted_rectifiedL, undistorted_rectifiedR)  # .astype(np.float32)/16
			dispr = right_matcher.compute(undistorted_rectifiedR, undistorted_rectifiedL)  # .astype(np.float32)/16

			self.Q = self.Q.astype(np.float32) 
			displ = np.int16(displ)
			dispr = np.int16(dispr)
			filteredImg = wls_filter.filter(displ, undistorted_rectifiedL, None, dispr)  # important to put "imgL" here!!!

			filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
			filteredImg = np.uint8(filteredImg)

			
			disparity = left_matcher.compute(undistorted_rectifiedL, undistorted_rectifiedR).astype(np.float32) / 16.0

			
			points = cv2.reprojectImageTo3D(disparity, self.Q)
			
			#reflect on x axis
			reflect_matrix = np.identity(3)
			reflect_matrix[0] *= -1
			points = np.matmul(points,reflect_matrix)

			#extract colors from image
			
			colors = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)

			#filter by min disparity
			mask = disparity > disparity.min()
			out_points = points[mask]
			out_colors = colors[mask]


			write_ply('out.ply', out_points, out_colors)
			print('%s saved' % 'out.ply')


			cv2.imshow(self.windowNameL,undistorted_rectifiedL)
			cv2.imshow(self.windowNameR,undistorted_rectifiedR)

			if (apply_colourmap):
			
				disparity_colour_mapped = cv2.applyColorMap((filteredImg * (256. / 160)).astype(np.uint8), cv2.COLORMAP_RAINBOW )
				cv2.imshow(windowNameD, disparity_colour_mapped)
			else:

				cv2.imshow(windowNameD, filteredImg)
			
			
			# start the event loop - essential
			
			key = cv2.waitKey(int(1000/self.disparity_processing_framerate)) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)

			# It can also be set to detect specific key strokes by recording which key is pressed

			# e.g. if user presses "x" then exit

			if (key == ord(' ')):
				keep_processing = False
			elif (key == ord('c')):
				apply_colourmap = not(apply_colourmap)

			elif (key == ord('x')):
				exit()
	def getStereoCalibrationImages(self):
		i = 0
		for idx in range(1, self.maxImageCount+1):
			
			frameL = cv2.imread('calibImage/ImageL%02d.jpg' %idx) 
			if frameL is not None :
				i = i+1
				self.frameL.append(frameL)
			frameR = cv2.imread('calibImage/ImageR%02d.jpg' %idx)
			if frameR is not None :
				self.frameR.append(frameR)

		self.maxImageCount = i

def main():
	zed = ZED()
	#if we already have a calibration file(we did run the code wth CalibrationFile is False parameter before)
	#set this parameter to True and no need to calibrate the camera for every run...

	if zed.CalibrationFile is False:
		#after capturing all corners , save it to a folder, eliminate some of them manually and then reload them from file
		if zed.fromFile is True:
			zed.getCalibrationImages()
		else:
			frameL, frameR = zed.AcquireFrames()

		zed.DisplayFrames(frameL, frameR )
		grayL, grayR = zed.FindImagePoints()
		zed.Calibrate(grayL, grayR)
		zed.CalculateReprojectionError()
		zed.getStereoCalibrationImages()
		zed.ExtrinsicCalibration()
		zed.RectifyImages()
	
	zed.CalculateStereoDepth()

if __name__ == '__main__':
    main()