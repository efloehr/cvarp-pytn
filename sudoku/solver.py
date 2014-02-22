from SimpleCV import Image
import cv2
import numpy as np
import sudoku


# Read the stored digits
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))
model = cv2.KNearest()
model.train(samples,responses)


def get_puzzle_from_image(raw_image):
    # Returns None if no puzzle found
    # Returns puzzle, x offset, y offset
    # if puzzle found. Offsets are top
    # left corner of puzzle

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
    if puzzle_blob is None: return None, 0, 0

    # Crop image to just the puzzle
    puzzle_image = puzzle_blob.crop()
    offset_x, offset_y = puzzle_blob.topLeftCorner()

    return puzzle_image, offset_x, offset_y


def read_puzzle_from_image(raw_image, puzzle_image, puzzle_offset_x, puzzle_offset_y):
    # Return puzzle representation, dictionary of image pictures

    # Slice into 81 squares (simple approach)
    blocks = []
    block_width = puzzle_image.width / 9.0
    block_height = puzzle_image.height / 9.0

    if block_width<10 or block_height<10:
        return '',[]

    for y in range(0,9):
        for x in range(0,9):
            block = puzzle_image.crop(x*block_width, y*block_height, block_width, block_height, smart=True)
            blocks.append((block, x*block_width, y*block_height))

    puzzle_repr = ""
    number_pictures = {}

    # For each of the 81 blocks, find the number, if any
    global model # KNearest model
    for block, block_offset_x, block_offset_y in blocks:
        blobblocks = block.findBlobs()
        digit = '.'

        # For each feature found in the block, see if it is a number
        if blobblocks is None: continue
        for blobblock in blobblocks:
            blob_offset_x, blob_offset_y = blobblock.topLeftCorner()
            blockwidthpct = blobblock.width() / block_width
            blockheightpct = blobblock.height() / block_height

            # This is empirically determined and could be improved
            if blockwidthpct > .2 and blockwidthpct < .9:
                if blockheightpct > .2 and blockheightpct < .9:

                    # Set the found number to a standard size for comparison
                    blobsmall = blobblock.blobImage().resize(10,10).getGrayNumpyCv2()
                    blobsmall = blobsmall.reshape((1,100))
                    blobsmall = np.float32(blobsmall)

                    # Get the number that matches closest to the training set
                    retval, results, neigh_resp, dists = model.find_nearest(blobsmall, k = 1)
                    digit = str(int((results[0][0])))

                    # Grab the number blob from the original image for pasting
                    # as solved numbers into the puzzle
                    digit_x = puzzle_offset_x + block_offset_x + blob_offset_x
                    digit_y = puzzle_offset_y + block_offset_y + blob_offset_y
                    number_pictures[digit] = raw_image.crop(digit_x, digit_y, blobblock.width(), blobblock.height(), smart=True)

        # Will be . if a number wasn't found, else the found digit
        puzzle_repr += digit
    
    # Return the string that represents the seen puzzle
    return puzzle_repr, number_pictures


def augment_image_with_solution(raw_image, puzzle_offset_x, puzzle_offset_y, puzzle_width, puzzle_height, puzzle_repr, solved_puzzle, number_pictures):
    block_width = int(round(puzzle_width / 9.0))
    block_height = int(round(puzzle_height / 9.0))
    # For each of the 81 blocks, paste the solved number on empty blocks
    for y in range(0,9):
        for x in range(0,9):
            orig_empty = puzzle_repr[y*9 + x] == '.'
            if not orig_empty: continue
            rowcol = chr(65+y) + str(x+1)
            value = solved_puzzle[rowcol]
            numpic = number_pictures[value]
            xpaste = int(round(puzzle_offset_x + (block_width * (x+0.5)) - (numpic.width/2.0)))
            ypaste = int(round(puzzle_offset_y + (block_height * (y+0.5)) - (numpic.height/2.0)))
            raw_image.dl().blit(numpic, (xpaste, ypaste))

def solve(raw_image):
    # Take a raw image and return an augmented image if puzzle can be solved
    # Find the puzzle and paste on to raw_image
    puzzle_image, puzzle_offset_x, puzzle_offset_y = get_puzzle_from_image(raw_image)

    if puzzle_image is not None:
        # Recognize the numbers
        puzzle_repr, number_pictures = read_puzzle_from_image(raw_image, puzzle_image,
                                                              puzzle_offset_x, puzzle_offset_y)

        # If there are less than 17 numbers, we can't solve
        # or if there aren't 81 blocks
        print puzzle_repr
        if len(puzzle_repr) <> 81 or puzzle_repr.count('.') > 64:
            return

        # Solve the puzzle
        solved_puzzle = sudoku.solve(puzzle_repr)
        print solved_puzzle
        if not solved_puzzle:
            return

        print solved_puzzle
        # Place the solution on the original image and return
        augment_image_with_solution(raw_image, puzzle_offset_x, puzzle_offset_y,
                                    puzzle_image.width, puzzle_image.height, 
                                    puzzle_repr, solved_puzzle, number_pictures)

# Test
if __name__ == '__main__':
    # Load the Image
    raw_image = Image('images/sudoku.jpg')

    solve(raw_image)
    raw_image.show()
    raw_input()

