import numpy as np
import cv2


def svertka_CV(inp_image, inp_kernel):
    kernel = inp_kernel / (np.sum(inp_kernel) if np.sum(inp_kernel) != 0 else 1)
    Kernel_result = cv2.filter2D(inp_image, -1, kernel)

    return Kernel_result


def svertka_no_CV(inp_image, inp_kernel):
    kernel_length = inp_kernel.shape[0]
    kernel_center = kernel_length // 2

    height, width, color = inp_image.shape
    matrix_w = width + kernel_center * 2
    matrix_h = height + kernel_center * 2
    matrix = np.array([[[0] * color] * matrix_w] * matrix_h)

    for c in range(0, color):
        for h in range(0, height):
            for w in range(0, width):
                matrix[h + kernel_center, w + kernel_center, c] = inp_image[h, w, c]
    for c in range(0, color):
        for h in range(0, kernel_center):
            for w in range(0, width):
                matrix[h, kernel_center + w, c] = inp_image[kernel_center - h, w, c]

    for c in range(0, color):
        for h in range(0, kernel_center):
            for w in range(0, width):
                matrix[matrix_h - h - 1, kernel_center + w, c] = inp_image[height - kernel_center + h - 1, w, c]

    for c in range(0, color):
        for h in range(0, height):
            for w in range(0, kernel_center):
                matrix[kernel_center + h, w, c] = inp_image[h, kernel_center - w, c]

    for c in range(0, color):
        for h in range(0, height):
            for w in range(0, kernel_center):
                matrix[kernel_center + h, matrix_w - w - 1, c] = inp_image[h, width - kernel_center + w - 1, c]

    height, width, color = matrix.shape
    image_result = matrix.copy()

    for h in range(kernel_center, height - kernel_center):
        for w in range(kernel_center, width - kernel_center):
            for c in range(0, color):
                start_height = h - kernel_center
                finish_height = start_height + kernel_length
                start_width = w - kernel_center
                finish_width = start_width + kernel_length

                mult_matrix = matrix[start_height:finish_height, start_width:finish_width, c] * inp_kernel
                kernel_sum = np.sum(inp_kernel)
                if kernel_sum != 0:
                    result = int(np.sum(mult_matrix) / np.sum(inp_kernel))
                else:
                    result = np.sum(mult_matrix)

                if result < 0:
                    image_result[h, w, c] = 0
                elif result > 255:
                    image_result[h, w, c] = 255
                else:
                    image_result[h, w, c] = result

    return image_result[kernel_center:-kernel_center, kernel_center: -kernel_center, :].astype(np.uint8)


kernel_0 = np.array([[40, 2, -20, 2, 40],
                     [2, 2, 2, 2, 2],
                     [-20, 2, 40, 2, -20],
                     [2, 2, 2, 2, 2],
                     [40, 2, -20, 2, 40]])

kernel_1 = np.array([[0, -2, 0],
                     [-2, 8, -2],
                     [0, -2, 0]])

kernel_2 = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

kernel_3 = np.array([[-5, -5, -5],
                     [-5, 45, -5],
                     [-5, -5, -5]])

kernel_4 = np.array([[-2.0, 4.0, -2.0],
                     [0, 0, 0],
                     [-2.0, 4.0, -2.0]])

kernel_5 = np.array([[4, 8, 4],
                     [0, 0, 0],
                     [-4, -8, -4]])

kernels = (("Blur and Sharpen", kernel_0),
           ("High Pass", kernel_1),
           ("High Pass inv", kernel_2),
           ("Sharpen", kernel_3),
           ("Detect Vertical", kernel_4),
           ("Horizontal strange", kernel_5))

Source_image = cv2.imread('test.jpg')
cv2.imshow('Source_image', cv2.resize(Source_image, (400, 400)))

for kernel_Name, kernel in kernels:
    print('It`s still working, wait plz))))')
    cv2.waitKey(1)
    Edited_image_cv = svertka_CV(Source_image, kernel)
    Edited_image_no_cv = svertka_no_CV(Source_image, kernel)
    cv2.imshow("*** {} kernel | OpenCV ***".format(kernel_Name), cv2.resize(Edited_image_cv, (400, 400)))
    cv2.imshow("*** {} kernel | no_CV ***".format(kernel_Name), cv2.resize(Edited_image_no_cv, (400, 400)))

print('That`s the end))))')
cv2.waitKey(0)
cv2.destroyAllWindows()
