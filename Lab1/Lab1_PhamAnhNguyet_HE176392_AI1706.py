
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button, TextBox
import math 

def onselect(eclick, erelease):
    """Callback function to store the coordinates of the selected rectangle"""
    global x1, y1, x2, y2, rect
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                    linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.draw()
    print(f"Selected rectangle: ({x1}, {y1}), ({x2}, {y2})")
    print(rect)
    

def translate(event):
    """Callback function to translate the selected rectangle"""
    global rect
    if rect is None:
        print("No rectangle selected.")
        return
    try:
        dx, dy = [float(_) for _ in text_box.text.split(',')]
        rect.set_xy((rect.get_x(), rect.get_y() ))
        fig.canvas.draw_idle()
        result_t = plt.Rectangle((x1 + dx, y1 + dy), (x2-x1), (y2-y1),
                    linewidth=1, edgecolor='m', facecolor='none')
        ax.add_patch(result_t)
        plt.draw()
    except ValueError:
        print("Invalid input.")


def rotate(event):
    """Callback function to translate the rotated rectangle"""
    global rect
    if rect is None:
        print("No rectangle selected.")
        return
    try:
        angle_degree = float(text_box2.text)
        angle = math.radians(angle_degree)
        center = ((x1 + (x2-x1)/2), (y1+ (y2-y1)/2))
        point = (x1,y1)
        x_ = (point[0] - center[0]) * math.cos(angle) - \
        (point[1] - center[1]) * math.sin(angle) + center[0]
        y_ = (point[0] - center[0]) * math.sin(angle) + \
        (point[1] - center[1]) * math.cos(angle) + center[1]
        # rect.set_xy((rect.get_x() + dx, rect.get_y() + dy))
        fig.canvas.draw_idle()
        result_t = plt.Rectangle((x_,y_), x2-x1, y2-y1,angle= angle_degree,
                    linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(result_t)
        plt.draw()
    except ValueError:
        print("Invalid input.")

def scale(event):
    """Callback function to translate the scaled rectangle"""
    global rect
    if rect is None:
        print("No rectangle selected.")
        return
    try:
        dx, dy = [float(_) for _ in text_box3.text.split(',')]
        rect.set_xy((rect.get_x(), rect.get_y()))
        fig.canvas.draw_idle()
        result_t = plt.Rectangle((x1*dx, y1*dy), (x2-x1)*dx, (y2-y1)*dy,
                    linewidth=1, edgecolor='c', facecolor='none')
        ax.add_patch(result_t)
        plt.draw()
    except ValueError:
        print("Invalid input.")

fig, ax = plt.subplots()
ax.set_xlim(0, 30)
ax.set_ylim(0, 20)
plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin to make room for the button
ax.grid(True)


# Create a RectangleSelector object
selector = RectangleSelector(ax, onselect)

# TRANSLATE
# Create a Button object
button_t = Button(plt.axes([0.1, 0.125, 0.2, 0.05]), "Translate")
button_t.on_clicked(translate)

# Create a TextBox object
text_box = TextBox(plt.axes([0.1, 0.05, 0.2, 0.05]), "tx, ty: ", initial="")

# ROTATE
# Create a Button object
button_r = Button(plt.axes([0.4, 0.125, 0.2, 0.05]), "Rotate")
button_r.on_clicked(rotate)

# Create a TextBox object
text_box2 = TextBox(plt.axes([0.4, 0.05, 0.2, 0.05]), "angle: ", initial="")

# Scale
# Create a Button object
button_s = Button(plt.axes([0.7, 0.125, 0.2, 0.05]), "Scale")
button_s.on_clicked(scale)

# Create a TextBox object
text_box3 = TextBox(plt.axes([0.7, 0.05, 0.2, 0.05]), "sx, sy: ", initial="")

# # reset
# def reset(event):
#     ax.cla()
#     ax.set_xlim(0, 30)
#     ax.set_ylim(0, 20)
#     ax.grid(True)
#     rect = None

# button_re = Button(plt.axes([0.4, 0.9, 0.2, 0.05]), "Reset")
# button_re.on_clicked(reset)
# Initialize the rectangle variable
rect = None

plt.show()