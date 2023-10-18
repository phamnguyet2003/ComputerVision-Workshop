import tkinter as tk
from PIL import ImageTk, Image

root = tk.Tk()
root.geometry('400x300')

# Load the background image
bg_image = Image.open('eiffel.jpeg')
bg_image = bg_image.resize((400, 300), Image.ANTIALIAS)
bg_image_tk = ImageTk.PhotoImage(bg_image)

# Create a canvas and add the background image
canvas = tk.Canvas(root, width=400, height=300)
canvas.pack(fill='both', expand=True)
canvas.create_image(0, 0, image=bg_image_tk, anchor='nw')

# Add other widgets to the canvas
button = tk.Button(canvas, text='Click me!', font=('Arial', 14))
button_window = canvas.create_window(200, 150, anchor='center', window=button)

root.mainloop()
