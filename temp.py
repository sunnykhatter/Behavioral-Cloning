# #Creating existing image data 
# images = []
# measurements = []
# for line in lines:
# 		source_path_center = line[0] #Center, left, right
# 		source_path_right = line[1]
# 		source_path_left = line[2]
		
# 		image_center = cv2.imread(source_path_center)
# 		image_left = cv2.imread(source_path_left)
# 		image_right = cv2.imread(source_path_right)
		
# 		#Append Images to the training set
# 		images.append(image_center)
# 		images.append(image_left)
# 		images.append(image_right)


# 		steering_center = float(line[3])
# 		correction = 0.2
# 		steering_left = steering_center	+ correction
# 		steering_right = steering_center - correction

# 		measurements.append(steering_center)
# 		measurements.append(steering_left)
# 		measurements.append(steering_right)
		
# 		#Create augmented images of the existing data
# 		augImageCenter = np.fliplr(image_center)
# 		augMesasurment = -steering_center

# 		#Append the images to the training set
# 		images.append(augImageCenter)
# 		measurements.append(augMesasurment)

