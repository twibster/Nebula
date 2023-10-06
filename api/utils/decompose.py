import os

def decompose_img(image_path, output_path):
    image = cv.imread(image_path)

    blue_channel, green_channel, red_channel = cv.split(image)
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    zeros = np.zeros_like(blue_channel)
    cv.imwrite(os.path.join(output_path, 'red_channel_only.jpg'), cv.merge([zeros, zeros, red_channel]))
    cv.imwrite(os.path.join(output_path, 'green_channel_only.jpg'), cv.merge([zeros, green_channel, zeros]))
    cv.imwrite(os.path.join(output_path, 'blue_channel_only.jpg'), cv.merge([blue_channel, zeros, zeros]))
    cv.imwrite(os.path.join(output_path, 'grayscale_image.jpg'), gray_image)