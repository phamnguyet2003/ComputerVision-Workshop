import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageTk, Image
import cv2
import numpy as np

# im1, im2 are links of 2 images
global im1, im2
im1 = None
im2 = None

def resize(image, width):
    # Define the new size (width, height) for the image
    new_size = (width, int(image.shape[0] * (width / image.shape[1])))

    # Resize the image
    resized_image = cv2.resize(image, new_size)
    
    return resized_image

def on_button_clicked():
    global im1
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()
    if filename:
        im1 = filename
        print('You choose:', filename)

def on_button_clicked2():
    global im2
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename()
    if filename:
        im2 = filename
        print('You choose:', filename)

def match_imgs (link1, link2):
    img_template = plt.imread(link1)
    img_need_aligned = plt.imread(link2)
    img_template = resize(img_template, 700)
    img_need_aligned = resize(img_need_aligned, 700)
    im1 = img_template.copy()
    im2 = img_need_aligned.copy()

    MAX_FEATURES = 500
    im1Gray = cv2.cvtColor(img_need_aligned, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x:x.distance)

    # Remove not so good matches
    GOOD_MATCH_PERCENT = 0.2
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(img_need_aligned, keypoints1, img_template, keypoints2, matches, None)
    # fig, ax = plt.subplots(figsize = (10,10))
    # ax.imshow(imMatches)
    points1 = np.zeros((len(matches), 2), dtype="float")
    points2 = np.zeros((len(matches), 2), dtype="float")
    for i, match in enumerate(matches):
        points1[i] = keypoints1[match.queryIdx].pt
        points2[i] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, method = cv2.RANSAC)

    # Use homography
    (height, width) = img_template.shape[:2]
    im1Reg = cv2.warpPerspective(img_need_aligned, h, (width, height))
    # plt.imshow(im1Reg[:,:,::-1]);plt.title("Aligned")
    return imMatches, im1Reg



def on_button_clicked3():
    global im1, im2
    fig, ax = plt.subplots(1,1,figsize = (10,10))
    ax.imshow(match_imgs(im1, im2)[0])
    plt.show()
    
    

def on_button_clicked4():
    global im1, im2
    fig, ax = plt.subplots(1,1,figsize = (10,10))
    ax.imshow(match_imgs(im1, im2)[1])
    plt.show()

root = tk.Tk()
root.geometry('600x600')

# Create a canvas
canvas = tk.Canvas(root, width=400, height=300)
canvas.pack(fill='both', expand=True)

# Add a button to the canvas
browse_button = tk.Button(canvas, text='Browse Template', command=on_button_clicked, width=15, height = 1)
browse_button.pack(pady=200)
browse_button.place(x=350, y=525)

# Add a button to the canvas
browse_button2 = tk.Button(canvas, text='Browse Test', command=on_button_clicked2, width=15, height = 1)
browse_button2.pack(pady=200)
browse_button2.place(x=475, y=525)


# Load the background image
# bg_image = Image.open(r'D:\Nguyet Folder\CPV301\Lab5\eiffel.jpeg')
bg_image = Image.open('eiffel.jpeg')
bg_image = bg_image.resize((600, 500), Image.ANTIALIAS)
bg_image_tk = ImageTk.PhotoImage(bg_image)

# Add the background image to the canvas
canvas.create_image(0, 0, image=bg_image_tk, anchor='nw')
# Add some text to the canvas
text = 'Feature-Based Alignment \n RANSAC Algorithm'
canvas.create_text(200, 550, text=text, font=('Arial', 17), fill='black')

# Add a button to the canvas
browse_button3 = tk.Button(canvas, text='Matches', command=on_button_clicked3, width=15, height = 1)
browse_button3.pack(pady=200)
browse_button3.place(x=350, y=552)

# Add a button to the canvas
browse_button4 = tk.Button(canvas, text='Align', command=on_button_clicked4, width=15, height = 1)
browse_button4.pack(pady=200)
browse_button4.place(x=475, y=552)

def on_closing():
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()