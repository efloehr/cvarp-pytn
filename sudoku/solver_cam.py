from SimpleCV import Camera
from solver import solve

if __name__ == '__main__':
    while 1:
        cam = Camera()
        img = cam.getImage()
        solve(img)
        img.show()

