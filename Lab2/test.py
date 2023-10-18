import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load the image
img = plt.imread('cat.jpg')

# Display the image
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)
im = ax.imshow(img)

# Set up the sliders for brightness and contrast
ax_brightness = plt.axes([0.5, 0.25, 0.4, 0.03])
s_brightness = Slider(ax_brightness, 'Brightness', -1.0, 1.0, valinit=0.0, color='#9400D3')

ax_contrast = plt.axes([0.5, 0.3, 0.4, 0.03])
s_contrast = Slider(ax_contrast, 'Contrast', 0, 2.0, valinit=1.0, color='#3CB371')

# Set up the sliders for red, green, and blue channels
ax_r = plt.axes([0.2, 0.05, 0.7, 0.03])
s_r = Slider(ax_r, 'Red ', 0.0, 1.0, valinit=0, color='red')

ax_g = plt.axes([0.2, 0.1, 0.7, 0.03])
s_g = Slider(ax_g, 'Green ', 0, 1.0, valinit=0, color='green')

ax_b = plt.axes([0.2, 0.15, 0.7, 0.03])
s_b = Slider(ax_b, 'Blue ', 0, 1.0, valinit=0, color='blue')

# Global variable to store the original image
original_img = img.copy()

# Function to update the image when the brightness or contrast sliders change
def update_brightness_contrast(val):
    global img

    # Get the current brightness and contrast values
    brightness = s_brightness.val
    contrast = s_contrast.val

    # Update the image with the new brightness and contrast values
    img = original_img * contrast + brightness
    img = np.clip(img, 0, 1)

    # Update the image display
    im.set_data(img)
    fig.canvas.draw_idle()

    # Calculate the new brightness and contrast values after adjusting the color channels
    r_val = s_r.val
    g_val = s_g.val
    b_val = s_b.val

    new_brightness = brightness - (r_val * 0.299 + g_val * 0.587 + b_val * 0.114)
    new_contrast = contrast / (1 + (r_val * 0.299 + g_val * 0.587 + b_val * 0.114))

    # Update the slider values for brightness and contrast
    s_brightness.set_val(new_brightness)
    s_contrast.set_val(new_contrast)

# Function to update the image when the red, green, or blue channel sliders change
def update_color(val):
    try:
        # Get the current red, green, and blue channel values
        r_val = s_r.val
        g_val = s_g.val
        b_val = s_b.val

        # Create a copy of the original image to modify
        modified_img = original_img.copy()

        # Apply the adjustments to the color channels
        modified_img[:,:,0] += r_val
        modified_img[:,:,1] += g_val
        modified_img[:,:,2] += b_val

        # Update the image with the adjusted color channels
        im.set_data(modified_img)
        fig.canvas.draw_idle()

        # Calculate the new brightness and contrast values after adjusting the color channels
        new_brightness = s_brightness.val - (r_val * 0.299 + g_val * 0.587 + b_val * 0.114)
        new_contrast = s_contrast.val / (1 + (r_val * 0.299 + g_val * 0.587 + b_val * 0.114))

        # Update the slider values for brightness and contrast
        s_brightness.set_val(new_brightness)
        s_contrast.set_val(new_contrast)

    except ValueError:
        print("Invalid input.")

# Connect the sliders to their update functions
s_brightness.on_changed(update_brightness_contrast)
s_contrast.on_changed(update_brightness_contrast)
s_r.on_changed(update_color)
s_g.on_changed(update_color)
s_b.on_changed(update_color)

plt.show()
