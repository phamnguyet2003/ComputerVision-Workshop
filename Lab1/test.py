# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# from matplotlib.widgets import Button
# from matplotlib.widgets import TextBox
# import numpy as np
# import pandas as pd


# # fig,ax = plt.subplots()


# df = pd.read_csv('chart_data.csv', delimiter='\t')
# df = pd.DataFrame(df)
# print(df)

# periods = [str(_) for _ in df.index]
# issued = df['Issued']
# denied = df['Denied']
# pending = df['Pending']

# fig, ax = plt.subplots(figsize=(9,6))
# fig.subplots_adjust(bottom = 0.2)
# # ax.grid(True)
# ax.plot(periods, issued)
# ax.set_title(issued.name)

# plt.show()

############################
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Button
# from matplotlib.widgets import TextBox
# from matplotlib.widgets import RectangleSelector

# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin to make room for the button
# ax.grid(True)

# def reset():
#     global ax
#     ax.cla()
#     # fig, ax = plt.subplots()
#     plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin to make room for the button
#     ax.grid(True)
#     rect = None


# ###############
# # DRAW BY MOUSE
# rect = None
# def onselect(eclick, erelease):
#     global rect
#     # Get the coordinates of the starting and ending points of the rectangle
#     xstart, ystart = eclick.xdata, eclick.ydata
#     xend, yend = erelease.xdata, erelease.ydata
    
#     # Draw a rectangle on the plot
#     rect = plt.Rectangle((xstart, ystart), xend - xstart, yend - ystart,
#                           linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#     # print(rect)
#     # rect = (xstart, ystart, xend - xstart, yend - ystart)
#     # print(rect)
#     plt.draw()

# # Create a plot and enable rectangle selection
# # fig, ax = plt.subplots()
# # ax.plot(x, y)
# selector = RectangleSelector(ax, onselect)

# ##################
# # TEXTBOX
# graphBox1 = fig.add_axes([0.1, 0.05, 0.2, 0.05])
# graphBox2 = fig.add_axes([0.4, 0.05, 0.2, 0.05])
# graphBox3 = fig.add_axes([0.7, 0.05, 0.2, 0.05])


# # text_box.on_submit(submit)
# text_box1 = TextBox(graphBox1, 'tx, ty:', initial="")
# text_box2 = TextBox(graphBox2, 'angle:', initial="")
# text_box3 = TextBox(graphBox3, 'sx, sy:', initial="")

# # Define a callback function for each textbox
# def submit_t(text):
#     global rect,ax
#     # print(rect)

#     tx, ty = [float(_) for _ in text.split(',')]
#     # print("Translate: ", tx, ty)

#     rect = plt.Rectangle((rect[0], rect[1]),rect[2],rect[3],
#                     linewidth=1, edgecolor='b', facecolor='none')
#     ax.add_patch(rect)
#     plt.draw()
# def translate(event):
#     """Callback function to translate the selected rectangle"""
#     global rect
#     if rect is None:
#         print("No rectangle selected.")
#         return
#     try:
#         dx, dy = float(text_box.text), 0.0
#         rect.set_xy((rect.get_x() + dx, rect.get_y() + dy))
#         fig.canvas.draw_idle()
#         result_t = plt.Rectangle((x1, y1), x2-x1, y2-y1,
#                     linewidth=1, edgecolor='b', facecolor='none')
#         ax.add_patch(result_t)
#         plt.draw()
#     except ValueError:
#         print("Invalid input.")

# def submit_r(text):
#     print("Rotate: " + text)

# def submit_s(text):
#     sx, sy = [int(_) for _ in text.split(',')]
#     # print("Scale: ", sx)

# #################
# # BUTTON

# def click_t(event):
#     global text_box1
#     text_box1.on_submit(submit_t)

# def click_r(event):
#     print('Rotation')
# def click_s(event):
#     print('Scaling')

# # Translate
# button_translation = plt.axes([0.1, 0.125, 0.2, 0.05])  # (left, bottom, width, height)
# button_t = Button(button_translation, 'Translate')
# button_t.on_clicked(text_box1.on_submit(submit_t))


# # Rotate
# button_rotation = plt.axes([0.4, 0.125, 0.2, 0.05])  # (left, bottom, width, height)
# button_r = Button(button_rotation, 'Rotation')
# button_r.on_clicked(click_r)

# # Scale
# button_scaling = plt.axes([0.7, 0.125, 0.2, 0.05])  # (left, bottom, width, height)
# button_s = Button(button_scaling, 'Scaling')
# button_s.on_clicked(click_s)

# # Reset
# button_reset = plt.axes([0.4, 0.9, 0.2, 0.05])  # (left, bottom, width, height)
# button_re = Button(button_reset, 'Reset')
# button_re.on_clicked(reset())

# # Set the callback functions for each textbox
# # text_box1.on_submit(submit_t)
# text_box2.on_submit(submit_r)
# text_box3.on_submit(submit_s)

# # Show the plot
# plt.show()


# import matplotlib.pyplot as plt
# from matplotlib.widgets import RectangleSelector, Button, TextBox
# import math 

# def onselect(eclick, erelease):
#     """Callback function to store the coordinates of the selected rectangle"""
#     global x1, y1, x2, y2, rect
#     x1, y1 = eclick.xdata, eclick.ydata
#     x2, y2 = erelease.xdata, erelease.ydata
#     rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
#                     linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#     plt.draw()
#     print(f"Selected rectangle: ({x1}, {y1}), ({x2}, {y2})")
#     print(rect)
    

# def translate(event):
#     """Callback function to translate the selected rectangle"""
#     global rect
#     if rect is None:
#         print("No rectangle selected.")
#         return
#     try:
#         dx, dy = [float(_) for _ in text_box.text.split(',')]
#         rect.set_xy((rect.get_x() + dx, rect.get_y() + dy))
#         fig.canvas.draw_idle()
#         result_t = plt.Rectangle((x1, y1), (x2-x1), (y2-y1),
#                     linewidth=1, edgecolor='b', facecolor='none')
#         ax.add_patch(result_t)
#         plt.draw()
#     except ValueError:
#         print("Invalid input.")


# def rotate(event):
#     """Callback function to translate the selected rectangle"""
#     global rect
#     if rect is None:
#         print("No rectangle selected.")
#         return
#     try:
#         angel = float(text_box2.text)
#         # rect.set_xy((rect.get_x() + dx, rect.get_y() + dy))
#         fig.canvas.draw_idle()
#         result_t = plt.Rectangle((x1, y1), x2-x1, y2-y1,angle= angel,
#                     linewidth=1, edgecolor='b', facecolor='none')
#         ax.add_patch(result_t)
#         plt.draw()
#     except ValueError:
#         print("Invalid input.")

# def scale(event):
#     """Callback function to translate the selected rectangle"""
#     global rect
#     if rect is None:
#         print("No rectangle selected.")
#         return
#     try:
#         dx, dy = [float(_) for _ in text_box3.text.split(',')]
#         rect.set_xy((rect.get_x(), rect.get_y()))
#         fig.canvas.draw_idle()
#         result_t = plt.Rectangle((x1*dx, y1*dy), (x2-x1)*dx, (y2-y1)*dy,
#                     linewidth=1, edgecolor='b', facecolor='none')
#         ax.add_patch(result_t)
#         plt.draw()
#     except ValueError:
#         print("Invalid input.")

# fig, ax = plt.subplots()
# ax.set_xlim(0, 30)
# ax.set_ylim(0, 20)
# plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin to make room for the button
# ax.grid(True)


# # Create a RectangleSelector object
# selector = RectangleSelector(ax, onselect)

# # TRANSLATE
# # Create a Button object
# button_t = Button(plt.axes([0.1, 0.125, 0.2, 0.05]), "Translate")
# button_t.on_clicked(translate)

# # Create a TextBox object
# text_box = TextBox(plt.axes([0.1, 0.05, 0.2, 0.05]), "tx, ty: ", initial="")

# # ROTATE
# # Create a Button object
# button_r = Button(plt.axes([0.4, 0.125, 0.2, 0.05]), "Rotate")
# button_r.on_clicked(rotate)

# # Create a TextBox object
# text_box2 = TextBox(plt.axes([0.4, 0.05, 0.2, 0.05]), "angle: ", initial="")

# # Scale
# # Create a Button object
# button_s = Button(plt.axes([0.7, 0.125, 0.2, 0.05]), "Scale")
# button_s.on_clicked(scale)

# # Create a TextBox object
# text_box3 = TextBox(plt.axes([0.7, 0.05, 0.2, 0.05]), "sx, sy: ", initial="")

# # Initialize the rectangle variable
# rect = None

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load image
img = plt.imread('your_image.jpg')

# Create figure and axis
fig, ax = plt.subplots()

# Show image
im = ax.imshow(img)

# Create slider for brightness adjustment
ax_brightness = plt.axes([0.1, 0.05, 0.8, 0.05])
s_brightness = Slider(ax_brightness, 'Brightness', -1.0, 1.0, valinit=0, valstep=0.05)


def update(val):
    # Get current brightness value from slider
    brightness = s_brightness.val
    
    # Apply brightness adjustment to image
    modified_img = img + brightness
    
    # Update image data
    im.set_data(modified_img)
    
    # Draw canvas
    fig.canvas.draw_idle()
    
    # Print RGB values of first pixel in original and modified images
    print("Original image (first pixel): R={}, G={}, B={}".format(
        img[0, 0, 0], img[0, 0, 1], img[0, 0, 2]))
    print("Modified image (first pixel): R={}, G={}, B={}".format(
        modified_img[0, 0, 0], modified_img[0, 0, 1], modified_img[0, 0, 2]))


# Connect slider to update function
s_brightness.on_changed(update)

# Show plot
plt.show()




