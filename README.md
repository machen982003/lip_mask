# lip_contour

Title: 

Lip make up

Team Members: 

Chen Ma

Summary: 

The aim of this project is to implement an algorithm for real-time lip detection and tracking and develop an iOS app that can apply lipstick of the selected color to the users lip using the front camera as a mirror.

Introduction
This lip make up app is very suit for the mobile vision app since it take advantage of the convenient of mobile devices such as easy to access and have built in functions in IOS. It is challenge since it used both the CPU and GPU, we need to extract data from GPU and do rest of work. So although I can do the algorithm part in OpenCV, I need to do more prepare work to extract the data.

Background
The Smerk provided the functions to detect human face and mouth area, which is the crucial step for this project. Besides, this framework do the detections in GPU, make it really fast and power saving.
With the opencv frame work, I can do the rest processes to extract the lip. 
The Saeed’s paper help to change the image from rgb space to a YIQ space, where mouth can be easily segement out in Q space.

Approach:
Get the area of mouth
	Use the Smerk to get mouth position, then use face size to calculate mouth bound box.

	CGRect mouth_r = CGRectMake (mouth_p.x-face.size.width*0.05, mouth_p.y - face.size.height*0.35, face.size.width*0.5, face.size.height*0.7);

Extract matrix of mouth area

	Use the GPU image extraction method to get the whole image first, then clone the mouth part into a opencv mat, mouth_rgb.

Change the mouth area into YIQ space

	Flatten the mouth_rgb to a 3*(cols*rows) matrix first, then multiply this matrix with a 3*3 matrix, change the image to YIQ space. 

Use threshold to segment the lip in Q matrix

	The result of Q matrix is presented below, we can see the lip area have higher value than the other parts. Then for a simpler method, I set a threshold for segement the lip. The threshold is set by 1.5*mean(Q)

Create mask for lip area, set lip area to the lip stick color

	For this part, I reshaped the Q matrix after threshold, and made it from GRAY into RGBA, so that I can mask this matrix on the mouth. 
	
Draw the mask on the mouth
	The result of drawing the mask is presented below.

References:
[1]Saeed U, Dugelay J L. Combining edge detection and region segmentation for lip contour extraction[M]//Articulated Motion and Deformable Objects. Springer Berlin Heidelberg, 2010: 11-20.

[2]Chan, T.F., Vese, L.A.: Active contours without edges. In: IEEE Transactions on Image Processing, vol.10, no.2, pp.266-277 (2001)


