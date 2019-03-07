from PIL import Image
import cv2
import numpy as np


class RemoveBackground(object):


    def __init__(self, input):
        # == Parameters =======================================================================
        BLUR = 21
        CANNY_THRESH_1 = 50
        CANNY_THRESH_2 = 150
        MASK_DILATE_ITER = 10
        MASK_ERODE_ITER = 10
        MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format

        #img = cv2.imread('C:/Temp/person.jpg')
        img=cv2.imread(input)
        #== Processing =======================================================================

        #-- Read image -----------------------------------------------------------------------

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


        #-- Edge detection -------------------------------------------------------------------
        edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        cv2.imshow('img_edges', edges)

        #-- Find contours in edges, sort by area ---------------------------------------------
        contour_info = []
        _, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Previously, for a previous version of cv2, this line was:
        #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Thanks to notes from commenters, I've updated the code but left this note
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]

        #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
        # Mask is black, polygon is white
        mask = np.zeros(edges.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))

        #-- Smooth mask, then blur it --------------------------------------------------------
        mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
        mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
        mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
        mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

        #-- Blend masked img into MASK_COLOR background --------------------------------------
        mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices,
        img         = img.astype('float32') / 255.0                 #  for easy blending

        masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
        masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit

        cv2.imshow('img_masked', masked)                                   # Display
        cv2.waitKey()

        #cv2.imwrite('C:/Temp/person-masked.jpg', masked)           # Save


class Poster(object):
    path = './img_templates/'
    p_image = 'wanted_dead_alive.jpg'
    test_face = 'test.jpg'
    RemoveBackground(path + test_face)


    poster = Image.open(path + p_image)
    face = Image.open(path + test_face)

    poster.paste(face, (160, 150))

    # Saved in the same relative location
    poster.save("pasted_picture.jpg")



if __name__ == "__main__":
    Poster()
