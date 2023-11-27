import cv2
import numpy as np

# Step 2: Load the two images
image1 = cv2.imread(
    "/media/fmiled/LaCie/owncloud/2023_CM_Fukuoka/2023_CM_Fukuoka_brasse_dames_50_finaleA/2023_CM_Fukuoka_brasse_dames_50_finaleA_fixeDroite.jpg"
)
image2 = cv2.imread(
    "/media/fmiled/LaCie/owncloud/2023_CM_Fukuoka/2023_CM_Fukuoka_brasse_dames_50_finaleA/2023_CM_Fukuoka_brasse_dames_50_finaleA_fixeGauche.jpg"
)

# Step 3: Convert the images to the RGB color space

# Step 4: Perform color correction
image1_lab = cv2.cvtColor(image1, cv2.COLOR_RGB2LAB)
image2_lab = cv2.cvtColor(image2, cv2.COLOR_RGB2LAB)

l1, a1, b1 = cv2.split(image1_lab)
l2, a2, b2 = cv2.split(image2_lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l2_eq = clahe.apply(l2)

image2_lab_eq = cv2.merge((l2_eq, a2, b2))
image2_eq = cv2.cvtColor(image2_lab_eq, cv2.COLOR_LAB2RGB)

# Step 5: Save the transformed image
cv2.imwrite("image2_transformed.jpg", image2_eq)
