import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import numpy as np
import cv2
import seaborn as sns
from PIL import Image

# Tạo hình ảnh ngẫu nhiên
img_path = 'cat.jpg'
img_o = plt.imread(img_path).astype(np.float32)/255.0
img = img_o.copy()

# Khởi tạo subplot và hiển thị hình ảnh ban đầu
fig, ax = plt.subplots()
plt.subplots_adjust(bottom = 0.55, top = 1)
im = ax.imshow(img)

# Tạo slider để điều chỉnh độ sáng, độ tương phản, ba kênh màu đỏ, xanh lá, xanh dương
ax_brightness = plt.axes([0.5, 0.25, 0.4, 0.03])
s_brightness = Slider(ax_brightness, 'Brightness', -1.0, 1.0, valinit=0.0, color='#9400D3')

ax_contrast = plt.axes([0.5, 0.3, 0.4, 0.03])
s_contrast = Slider(ax_contrast, 'Contrast', 0, 2.0, valinit=1.0, color='#3CB371')

# 0.1, 0.05, 0.2,
ax_r = plt.axes([0.2, 0.05, 0.7, 0.03])
s_r = Slider(ax_r, 'Red ', 0.0, 1.0, valinit=0, color='red')

ax_g = plt.axes([0.2, 0.1, 0.7, 0.03])
s_g = Slider(ax_g, 'Green ', 0, 1.0, valinit=0, color='green')

ax_b = plt.axes([0.2, 0.15, 0.7, 0.03])
s_b = Slider(ax_b, 'Blue ', 0, 1.0, valinit=0, color='blue')

# Hàm xử lý khi giá trị của slider thay đổi
def update_img(val):
    global img
    brightness = s_brightness.val
    contrast = s_contrast.val
    modified_img = img * contrast + brightness
    modified_img = np.clip(modified_img, 0, 1)
    im.set_data(modified_img)
    fig.canvas.draw_idle()

# Gọi hàm update khi giá trị của slider thay đổi
s_brightness.on_changed(update_img)
s_contrast.on_changed(update_img)

def update_color(val):
    try:
        r_val = s_r.val
        g_val = s_g.val
        b_val = s_b.val

        # Create a numpy array from the image
        img_arr = np.array(img)

        # Get the dimensions of the array
        height, width, channels = img_arr.shape

        # Loop through each pixel and adjust the color values
        for y in range(height):
            for x in range(width):
                r, g, b = img_arr[y][x]
                r += r_val
                g += g_val
                b += b_val
                img_arr[y][x] = [r, g, b]

        # Update the image with the adjusted color values
        ax.imshow(img_arr)
        # print(r_val)
        plt.draw()

    except ValueError:
        print("Invalid input.")

s_r.on_changed(update_color)
s_g.on_changed(update_color)
s_b.on_changed(update_color)

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.tick_params(axis = 'y', which = 'both', length = 0)

# Median
def median_filter(event):
    print('Median Filter')
    img_median = cv2.medianBlur(img, 5)

    fig , ax = plt.subplots(1,2, figsize = (10,10))
    ax[0].imshow(img_o)
    ax[1].imshow(img_median)
    ax[0].set_title("origin image")
    ax[1].set_title("Median filter")
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])
    ax[0].tick_params(axis = 'y', which = 'both', length = 0)
    ax[1].xaxis.set_ticks([])
    ax[1].yaxis.set_ticks([])
    ax[1].tick_params(axis = 'y', which = 'both', length = 0)
    plt.show()
# Create a histogram equalization
button_median = Button(plt.axes([0.1, 0.4, 0.2, 0.1]), "Median\nFilter")
button_median.on_clicked(median_filter)

# add noise

img_noise = img_o.copy()

def add_noise(img):
    row, col = img.shape[0], img.shape[1]

    num_of_noise = np.random.randint(300 , 10000)
    for i in range(num_of_noise):
        row_noise = np.random.randint(0, row -1)
        col_noise = np.random.randint(0, col - 1)
        
        img[row_noise][col_noise] = np.array([0,0,0])

    num_of_noise = np.random.randint(300, 10000)
    for i in range(num_of_noise):
        row_noise = np.random.randint(0, row -1)
        col_noise = np.random.randint(0, col - 1)
        
        img[row_noise][col_noise] = np.array([255,255,255])
    
    return img

# Mean
def mean_filter(event):
    print('Mean Filter')
    img = plt.imread(img_path)   
    
    kernel = np.array([ [1,1,1],
                        [1,1,1],
                        [1,1,1]])*1/9
    kernel_size = kernel.shape[0]

    # Lấy chiều cao và chiều rộng của ảnh
    height, width, channels = img.shape

    # Khởi tạo ma trận kết quả
    result = np.zeros_like(img)

    # Áp dụng Gaussian filter lên từng kênh màu
    for c in range(channels):
        for i in range(height-kernel_size+1):
            for j in range(width-kernel_size+1):
                result[i+2,j+2,c] = np.sum(np.multiply(img[i:i+kernel_size,j:j+kernel_size,c], kernel))


    fig , ax = plt.subplots(1,2, figsize = (10,10))
    ax[0].imshow(img_o)
    ax[1].imshow(result)
    ax[0].set_title("origin image")
    ax[1].set_title("Mean filter")
    ax[0].xaxis.set_ticks([])
    ax[0].yaxis.set_ticks([])
    ax[0].tick_params(axis = 'y', which = 'both', length = 0)
    ax[1].xaxis.set_ticks([])
    ax[1].yaxis.set_ticks([])
    ax[1].tick_params(axis = 'y', which = 'both', length = 0)
    plt.show()

# Create a histogram equalization
button_mean = Button(plt.axes([0.4, 0.4, 0.2, 0.1]), "Mean\nFilter")
button_mean.on_clicked(mean_filter)

# Gaussian Smoothing
def gaussian(event):

    print('Gaussian Smoothing')
    img_gau = img_o.copy()
    img = plt.imread(img_path)

    # Khởi tạo Gaussian kernel 2D (kích thước 5x5, sigma=1.5)
    kernel = np.array([ [1, 4, 7, 4,1],
                        [4,16,26,16,4],
                        [7,26,41,26,7],
                        [4,16,26,16,4],
                        [1, 4, 7, 4,1]],)*1/273
    kernel_size = kernel.shape[0]

    # Lấy chiều cao và chiều rộng của ảnh
    height, width, channels = img.shape

    # Khởi tạo ma trận kết quả
    result = np.zeros_like(img)

    # Áp dụng Gaussian filter lên từng kênh màu
    for c in range(channels):
        for i in range(height-kernel_size+1):
            for j in range(width-kernel_size+1):
                result[i+2,j+2,c] = np.sum(np.multiply(img[i:i+kernel_size,j:j+kernel_size,c], kernel))

    # Hiển thị ảnh gốc và ảnh đã lọc
    fig, ax = plt.subplots(1,2, figsize = (10,10))
    # ax[0].subplot(121)
    ax[0].imshow(img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    # ax[1].subplot(122)
    ax[1].imshow(result)
    ax[1].set_title('Gaussian Filtered Image')
    ax[1].axis('off')
    plt.show()

    
# Create a histogram equalization
button_gaussian = Button(plt.axes([0.7, 0.4, 0.2, 0.1]), "Gaussian\nSmoothing")
button_gaussian.on_clicked(gaussian)

# Histogram equalization


def his_equal(event):
    def equal_hist(hist):
        cumulator = np.zeros_like(hist, np.float64)
        for i in range(len(cumulator)):
            cumulator[i] = hist[:i].sum()
        #print(cumulator)
        new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
        #new_hist = np.uint8(new_hist)
        return new_hist

    def compute_hist(img):
        hist = np.zeros((256,), np.uint8)
        h, w = img.shape[:2]
        for i in range(h):
            for j in range(w):
                hist[img[i][j]] += 1
        return hist

    img_1 = cv2.imread(img_path)
    img_he = img_1.copy()
    img_he = cv2.cvtColor(img_he, cv2.COLOR_RGB2HSV)
    origin_v= img_he[:,:,2].reshape(1,-1)[0].copy()

    hist = compute_hist(img_he[:,:,2]).ravel()
    new_hist = equal_hist(hist)

    h, w = img_he.shape[:2]
    for i in range(h):
        for j in range(w):
            img_he[i,j,2] = new_hist[img_he[i,j,2]]

    after_v = img_he[:,:,2].reshape(1,-1)[0].copy()
    img_he = cv2.cvtColor(img_he, cv2.COLOR_HSV2RGB)
    img_he = cv2.cvtColor(img_he, cv2.COLOR_BGR2RGB)

    fig , ax = plt.subplots(2,2, figsize = (15,10))
    ax[0][0].imshow(img_o)
    ax[0][1].imshow(img_he)
    sns.histplot(origin_v,bins = 100,kde = True, color = 'red', ax = ax[1][0])
    sns.histplot(after_v,bins = 100,kde = True, color = 'green', ax = ax[1][1])
    ax[0][0].set_title("origin image")
    ax[0][1].set_title("histogram equalization image")
    ax[1][0].set_title("origin histogram")
    ax[1][1].set_title("histogram after equalization")
    plt.show()
       

    print('Histogram equalization')
    
# Create a histogram equalization
button_he = Button(plt.axes([0.1, 0.25, 0.2, 0.1]), "Histogram\nEqualization")
button_he.on_clicked(his_equal)

plt.show()