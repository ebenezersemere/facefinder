# libraries!
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import seaborn as sns

def opencv_open_into_rgb(image_file_name):
    """ open image_file_name and convert to rgb """
    image_raw = cv.imread(image_file_name, cv.IMREAD_COLOR)  # reads into BGR
    orig_num_rows, orig_num_cols, _ = image_raw.shape  # cool underscore variable!
    num_rows, num_cols, _ = image_raw.shape
    print(f"image with name {image_file_name} read with dimensions: {image_raw.shape}")

    # let's resize as long as any dimension is > 840...

    maxdim = max(num_rows, num_cols)
    while maxdim > 840:
        num_rows = num_rows // 2  # halve!
        num_cols = num_cols // 2  # halve!
        maxdim = max(num_rows, num_cols)  # try again...

    if orig_num_rows != num_rows:  # did we resize?
        print(f"resizing to rows, cols = {num_rows}, {num_cols}")
        image_raw = cv.resize(image_raw, dsize=(num_cols, num_rows), interpolation=cv.INTER_LINEAR)  # resizing

    # now, convert to rgb
    image_rgb = cv.cvtColor(image_raw, cv.COLOR_BGR2RGB)  # convert from BGR to RGB
    return image_rgb

filename = "smile_emoji.jpg"  # any image

image_rgb = opencv_open_into_rgb(filename)

fig, ax = plt.subplots(4,1, figsize=(6,16))  # this means ax will be a 1x4 numpy array of axes!

ax[0].imshow(image_rgb)                     # normal image
ax[0].axis('off')
ax[0].set_title('all colors')

ax[1].imshow(image_rgb[:,:,0],cmap="gray")  # red-channel image
ax[1].axis('off')
ax[1].set_title('red only')

ax[2].imshow(image_rgb[:,:,1],cmap="gray")  # green-channel image
ax[2].axis('off')
ax[2].set_title('green only')

ax[3].imshow(image_rgb[:,:,2],cmap="gray")  # blue-channel image
ax[3].axis('off')
ax[3].set_title('blue only')

plt.show()

def save_rgb_image( image_rgb, new_file_name ):
    """ saves the image  image_rgb  to the (string) name, new_file_name
        add the extension you'd like, png, jpg, ... it knows many (not all)
    """
    image_bgr = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)     # convert from BGR to RGB
    result = cv.imwrite(new_file_name,image_bgr)
    if result == True:
        print(f"image_rgb was saved to {new_file_name}")
    else:
        print(f"there was a problem saving image_rgb to {new_file_name}")

filename = "avengers.jpg"  # "coffee.jpg"  "flag.png"  "avengers.jpg" ... try others
image_rgb = opencv_open_into_rgb(filename)
save_rgb_image( image_rgb, "new_image.png" )  # it needs the extension

# Training

cascPath = "haarcascade_frontalface_default_2005.xml"
faceCascade = cv.CascadeClassifier(cascPath)

# Read the image
image_faces_rgb = opencv_open_into_rgb("smile_emoji.jpg")
image_faces_gray = cv.cvtColor(image_faces_rgb, cv.COLOR_RGB2GRAY)

fig, ax = plt.subplots()
ax.imshow(image_faces_gray,cmap="gray")
ax.axis('off')
ax.set_title('Gray')
plt.show()

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    image_faces_gray,       # this is the input image
    scaleFactor=1.05,       # this is the scale-resolution for detecting faces
    minNeighbors=1,         # this is how many nearby-detections are needed to ok a face
    minSize=(10,10),        # this is the minimum size for a face
    flags = cv.CASCADE_SCALE_IMAGE,   # (standard)
)
print(f"Found {len(faces)} faces!")

for i, face in enumerate(faces):
    x,y,w,h = face
    print(f"face {i}: {face}")

image_faces_drawn_rgb = image_faces_rgb.copy()  # copy onto which we draw the bounding boxes for the faces

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    # note that this draws on the color image!
    cv.rectangle(image_faces_drawn_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(f"Drawn! How does it do?")

LoFi = []  # list of Face images

for (x, y, w, h) in faces:
    # note that this draws on the color image!
    face = image_faces_rgb[y:y+h,x:x+h,:]  #, (x, y), (x+w, y+h), (0, 255, 0), 2)
    LoFi.append( face )

# show the image!
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(image_faces_drawn_rgb)
ax.axis('off')
ax.set_title('FACES???')
plt.show()

print(f"There are {len(LoFi)} faces detected - they are held in the list 'LoFi'")
print(f"Here are some of them...")
fig, ax = plt.subplots(3,3)  # this means ax will be a 3x3 numpy array of axes!
#ax[0,0].imshow(LoFi[0])
#ax[0,0].imshow(cv2.resize(LoFi[0],dsize=(20,20)))
LoFi[0] = LoFi[10]
LoFi[1] = LoFi[1]
ax[0,0].imshow(LoFi[0])
ax[0,0].axis('off')
ax[0,1].imshow(LoFi[1])
ax[0,1].axis('off')
ax[0,2].imshow(LoFi[2])
ax[0,2].axis('off')
ax[1,0].imshow(LoFi[3])
ax[1,0].axis('off')
ax[1,1].imshow(LoFi[4])
ax[1,1].axis('off')
ax[1,2].imshow(LoFi[5])
ax[1,2].axis('off')
ax[2,0].imshow(LoFi[6])
ax[2,0].axis('off')
ax[2,1].imshow(LoFi[7])
ax[2,1].axis('off')
ax[2,2].imshow(LoFi[8])
ax[2,2].axis('off')
plt.show()

LoLoFi = [ cv.resize(LoFiIm,dsize=(20,20)) for LoFiIm in LoFi ]
print(f"Now, all of the faces are at the same, low resolution")
fig, ax = plt.subplots(3,3)  # this means ax will be a 3x3 numpy array of axes!
#ax[0,0].imshow(LoFi[0])
#ax[0,0].imshow(cv2.resize(LoFi[0],dsize=(20,20)))
ax[0,0].imshow(LoLoFi[0])
ax[0,0].axis('off')
ax[0,1].imshow(LoLoFi[1])
ax[0,1].axis('off')
ax[0,2].imshow(LoLoFi[2])
ax[0,2].axis('off')
ax[1,0].imshow(LoLoFi[3])
ax[1,0].axis('off')
ax[1,1].imshow(LoLoFi[4])
ax[1,1].axis('off')
ax[1,2].imshow(LoLoFi[5])
ax[1,2].axis('off')
ax[2,0].imshow(LoLoFi[6])
ax[2,0].axis('off')
ax[2,1].imshow(LoLoFi[7])
ax[2,1].axis('off')
ax[2,2].imshow(LoLoFi[8])
ax[2,2].axis('off')
plt.show()


# Code to search for a specific face

A = np.zeros((9, 9))
for r in range(9):
    for c in range(9):
        res = cv.matchTemplate(LoLoFi[r], LoLoFi[c], cv.TM_SQDIFF_NORMED)
        # res is a 2d image, so... we extract the value
        A[r, c] = res[0, 0]

with np.printoptions(precision=3, suppress=True):  # suppresses exponential notation!
    print(f"{A}")

# heat map of correlations

sns.set_theme(style="white")
f, ax = plt.subplots(figsize=(7,5))
sns.heatmap(A, cmap="Purples_r")   # Purple heatmap because why not
ax.axis('off')
# ALTERNATIVELY:  cmap="Gray", vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

filename = "emoji_expressions.jpg"  # many faces!
image_rgb = opencv_open_into_rgb(filename)

fig, ax = plt.subplots(4,1, figsize=(6,16))  # this means ax will be a 1x4 numpy array of axes!

ax[0].imshow(image_rgb)                     # normal image
ax[0].axis('off')
ax[0].set_title('all colors')

ax[1].imshow(image_rgb[:,:,0],cmap="gray")  # red-channel image
ax[1].axis('off')
ax[1].set_title('red only')

ax[2].imshow(image_rgb[:,:,1],cmap="gray")  # green-channel image
ax[2].axis('off')
ax[2].set_title('green only')

ax[3].imshow(image_rgb[:,:,2],cmap="gray")  # blue-channel image
ax[3].axis('off')
ax[3].set_title('blue only')

plt.show()

# Read the image
image_faces_rgb = opencv_open_into_rgb("emoji_expressions.jpg")
image_faces_gray = cv.cvtColor(image_faces_rgb, cv.COLOR_RGB2GRAY)

fig, ax = plt.subplots()
ax.imshow(image_faces_gray,cmap="gray")
ax.axis('off')
ax.set_title('Gray')
plt.show()

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    image_faces_gray,       # this is the input image
    scaleFactor=1.05,       # this is the scale-resolution for detecting faces
    minNeighbors=1,         # this is how many nearby-detections are needed to ok a face
    minSize=(10,10),        # this is the minimum size for a face
    flags = cv.CASCADE_SCALE_IMAGE,   # (standard)
)
print(f"Found {len(faces)} faces!")

for i, face in enumerate(faces):
    x,y,w,h = face
    print(f"face {i}: {face}")

image_faces_drawn_rgb = image_faces_rgb.copy()  # copy onto which we draw the bounding boxes for the faces

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    # note that this draws on the color image!
    cv.rectangle(image_faces_drawn_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(f"Drawn! How does it do?")

LoFipr = []  # list of Face images

for (x, y, w, h) in faces:
    # note that this draws on the color image!
    face = image_faces_rgb[y:y+h,x:x+h,:]  #, (x, y), (x+w, y+h), (0, 255, 0), 2)
    LoFipr.append( face )

# show the image!
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(image_faces_drawn_rgb)
ax.axis('off')
ax.set_title('FACES???')
plt.show()

print(f"There are {len(LoFi)} faces detected - they are held in the list 'LoFi'")
print(f"Here are some of them...")
fig, ax = plt.subplots(3,3)  # this means ax will be a 3x3 numpy array of axes!
ax[0,0].imshow(LoFipr[0])
ax[0,0].axis('off')
ax[0,1].imshow(LoFipr[1])
ax[0,1].axis('off')
ax[0,2].imshow(LoFipr[2])
ax[0,2].axis('off')
ax[1,0].imshow(LoFipr[3])
ax[1,0].axis('off')
ax[1,1].imshow(LoFipr[4])
ax[1,1].axis('off')
ax[1,2].imshow(LoFipr[5])
ax[1,2].axis('off')
ax[2,0].imshow(LoFipr[6])
ax[2,0].axis('off')
ax[2,1].imshow(LoFipr[7])
ax[2,1].axis('off')
ax[2,2].imshow(LoFipr[8])
ax[2,2].axis('off')
plt.show()

