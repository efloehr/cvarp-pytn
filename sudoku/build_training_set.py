from SimpleCV import Image
import cv2
import numpy as np
from sys import argv

if __name__ == "__main__":
    image_file = argv[1]

    # Load the Image
    raw_image = Image(image_file)

    # Remove color
    gray_image = raw_image.grayscale()

    # Smooth to remove speckle
    smooth_image = gray_image.gaussianBlur((5,5),0)

    # Convert to Numpy Array For OpenCV use
    cv_image = smooth_image.getGrayNumpyCv2()

    # Adaptive threshold does much better than linear
    raw_thresh_image = cv2.adaptiveThreshold(cv_image,255,1,1,11,2)

    # Convert back to a SimpleCV image
    thresh_image = Image(raw_thresh_image)

    # For some reason it gets rotated and flipped, reverse
    thresh_image = thresh_image.rotate90().flipVertical()

    # Find "blobs" which are interesting items in the image
    blobs = thresh_image.findBlobs()

    # Assume the largest rectangular blob is our puzzle
    puzzle_blob = None
    puzzle_area = 0

    for blob in blobs:
        if blob.isRectangle() and blob.area() > puzzle_area:
            puzzle_blob = blob
            puzzle_area = blob.area()

    # Only continue if there is a puzzle
    #if puzzle_blob is None: return

    # Crop image to just the puzzle
    puzzle_image = puzzle_blob.crop()
    puzzle_image.save("puzzle_only.jpg")

    # Slice into 81 squares (simple approach)
    blocks = []
    block_width = puzzle_image.width / 9
    block_height = puzzle_image.height / 9

    for y in range(0,9):
        for x in range(0,9):
            block = puzzle_image.crop(x*block_width, y*block_height, block_width, block_height)
            blocks.append(block)

    # Find blobs in each square and ask what number it is (if any)
    samples =  np.empty((0,100))
    responses = []
    keys = '0123456789'

    for block in blocks:
        block_blobs = block.findBlobs()
        print "Found {0} blobs in block".format(len(block_blobs))
        for blob in block_blobs:
            blob.blobImage().show()
            key = raw_input()
            blobcv = blob.blobImage().getGrayNumpyCv2()
            blobcvsmall = cv2.resize(blobcv,(10,10))
            if key == '':
                continue
            else:
                responses.append(int(key))
                sample = blobcvsmall.reshape((1,100))
                samples = np.append(samples,sample,0)

    responses = np.array(responses,np.float32)
    responses = responses.reshape((responses.size,1))
    print "training complete"

    np.savetxt('generalsamples.data',samples)
    np.savetxt('generalresponses.data',responses)
