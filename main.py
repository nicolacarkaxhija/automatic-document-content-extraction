from __future__ import print_function
import cv2
import os.path
import numpy as np
from skimage.filters import threshold_sauvola
import math
import sys

MAX_MATCHES = 10000
GOOD_MATCH_PERCENT = 0.2


def content_extraction(img, template):
    # Suppress template pixels
    kernel = np.ones((7, 7), np.uint8)  # 5x5 nel paper
    # Extract content by subtracting the template dilation
    dilation = cv2.erode(template, kernel, iterations=1)
    img = np.maximum(img, 255 - dilation)

    # Global binarization using a 0.25 threshold
    threshold_percentage = 0.25
    threshold_graylevel = threshold_percentage * 255
    img = cv2.threshold(img, threshold_graylevel, 255, cv2.THRESH_BINARY)[1]

    return img


def image_alignment(img, template):
    # SIFT code has been moved in a non-free package
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_MATCHES)
    img_keypoints, img_descriptors = orb.detectAndCompute(img, None)
    template_keypoints, template_descriptors = orb.detectAndCompute(template, None)

    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(img_descriptors, template_descriptors, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove bad ones
    number_of_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:number_of_good_matches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = img_keypoints[match.queryIdx].pt
        points2[i, :] = template_keypoints[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Apply homography
    height, width = template.shape[:2]
    registered_img = cv2.warpPerspective(img, h, (width, height), borderValue=255)

    return registered_img


def binarize(img):
    threshold = threshold_sauvola(img, window_size=25)
    boolean_img = img > threshold
    binary_img = boolean_img.astype('uint8') * 255
    return binary_img


def sigmoid(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            # NIST and hand-filled invoices 96 and 196 respectively
            x[i][j] = (1 / (1 + math.exp(-(x[i][j] - 96)))) * 255
    return x


if __name__ == '__main__':

    dim = int(input("Insert number of partial models: "))

    path = "NIST"
    folder = os.listdir(path)
    number_files = dim * 2
    folder = folder[:number_files]

    sample_img = cv2.imread(path + "/" + folder.pop())
    height, width = sample_img.shape[:2]

    # Load binarized invoice images
    images = []
    for index, filename in enumerate(folder):
        img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        binary_img = binarize(img)
        images.append(binary_img)
        completion_percentage = (index + 1) / number_files * 100
        sys.stdout.write("\rLoading images: %d%%" % completion_percentage)
    sys.stdout.write("\rLoading images: 100%\n")

    # TEMPLATE GENERATION

    # Generate partial models
    # np.random.shuffle(images)
    partial_models = images[:dim]
    random_selection = np.copy(images[dim:])

    for index, img1 in enumerate(partial_models):
        print("Generating partial model no. " + str(index + 1))
        tmp = []

        for img2 in random_selection:
            # Aligning and intersecting images
            img2 = image_alignment(img2, img1)
            blurred2 = cv2.GaussianBlur(img2, ksize=(0, 0), sigmaX=1)
            blurred1 = cv2.GaussianBlur(img1, ksize=(0, 0), sigmaX=1)
            intersection = np.maximum(blurred1, blurred2)
            tmp.append(intersection)

        # Compute a partial model as average of aligned images
        avg = np.zeros((height, width))
        for img in tmp:
            avg = avg + img / len(tmp)
        avg = sigmoid(avg)
        partial_models[index] = avg.astype('uint8')

    # Align partial models to the first one
    for i in range(1, len(partial_models)):
        partial_models[i] = image_alignment(partial_models[i], partial_models[0])
    # Generate the template as average of partial models
    avg = np.zeros((height, width))
    for img in partial_models:
        avg = avg + img / len(partial_models)
    avg = sigmoid(avg)

    print("Saving template: template.png")
    cv2.imwrite("template.png", avg)

    # CONTENT EXTRACTION

    # Read reference image
    template_filename = "template.png"
    print("Reading template: ", template_filename)
    template = cv2.imread(template_filename, cv2.IMREAD_GRAYSCALE)

    # Read image to be aligned
    img_filename = "filled-tax-form.png"
    print("Reading incoming image: ", img_filename)
    img = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
    img = binarize(img)
    print("Aligning images ...")
    aligned_img = image_alignment(img, template)

    # Detect dynamic elements
    dynamic = content_extraction(aligned_img, template)
    # Merging detected regions using morphological opening
    # kernel = np.ones((5, 5), np.uint8)
    # dynamic = cv2.morphologyEx(dynamic, cv2.MORPH_OPEN, kernel)

    print("Saving extracted content: extracted-content.png")
    cv2.imwrite("extracted-content.png", dynamic)
