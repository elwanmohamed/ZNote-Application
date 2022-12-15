import base64
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request
import cv2
import os

app = Flask(__name__)


def saving_img_in_dir(file_path, img_in_bytes):
    file = open(file_path, 'wb')
    from_bytes_to_img = base64.b64decode(img_in_bytes)
    file.write(from_bytes_to_img)
    file.close()


def delete_img_from_dir(file_path):
    os.remove(file_path)


# def similarity_checker(img_path, images_dataset):
#     # query image
#     img1 = cv2.imread(img_path)
#     img1 = cv2.cvtColor(img1, 0)
#     sift = cv2.xfeatures2d.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     dis_images = np.full(len(images_dataset), 0)
#     perfect_mask = np.full(len(images_dataset), None)
#     masks = np.full(len(images_dataset), None)
#     images = images_dataset
#
#     for (i, imagePath) in zip(range(len(images_dataset)), images_dataset):
#         img2 = cv2.imread(imagePath)
#         kp2, des2 = sift.detectAndCompute(img2, None)
#         bf = cv2.BFMatcher()
#         matches = bf.match(des1, des2)
#         matches = sorted(matches, key=lambda x: x.distance)
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#         m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
#         matches_mask = mask.ravel().tolist()
#         sum, count = 0, 0
#         for match in matches:
#             if count >= 10:
#                 break
#             sum += match.distance
#             count += 1
#         dis_images[i] = sum
#         perfect_mask[i] = matches_mask
#         masks[i] = matches
#
#     return images[dis_images.argmin()]
#
#     # img2 = cv2.imread(images[dis_images.argmin()])  # trainImage
#     # kp2, des2 = sift.detectAndCompute(img2, None)
#     # # Draw first 10 matches.
#     # img3 = cv2.drawMatches(img1, kp1,
#     #                        img2, kp2,
#     #                        masks[dis_images.argmin()],
#     #                        flags=2, outImg=None)
#     # print("All Distances:")
#     # for i in range(len(dis_images)):
#     #     # image and = its distance
#     #     print("Image #" + str(images[i]) + " = " + str(dis_images[i]))
#     # print("Image with Minimum distance: " + str(images[dis_images.argmin()]))
#     #
#     # # never do this
#     # img4 = cv2.drawMatches(img1, kp1, img2, kp2,
#     #                        masks[dis_images.argmin()][:10],
#     #                        None,
#     #                        matchColor=(255, 0, 0),  # draw matches in green color
#     #                        matchesMask=perfect_mask[dis_images.argmin()][:10],  # draw only inliers
#     #                        flags=2)
#     # plt.imshow(img3, 'gray'), plt.show()
#     # plt.imshow(img4, 'gray'), plt.show()

def similarity_checker(img_path):
    count = 0
    images = []
    for a in os.listdir('images'):
        if a.lower().endswith(('.jpg', '.png', '.tif', '.tiff', '.gif', '.JPG')):
            count = count + 1
            images.append('images/' + a)
    # query image
    img1 = cv2.imread(img_path)
    img1 = cv2.cvtColor(img1, 0)
    sift = cv2.xfeatures2d.SIFT_create(150)
    kp1, des1 = sift.detectAndCompute(img1, None)
    dis_images = np.full(count, 0)
    perfect_mask = np.full(count, None)
    masks = np.full(count, None)

    for (i, a) in zip(range(len(images)), os.listdir('images')):
        if a.lower().endswith(('.jpg', '.png', '.tif', '.tiff', '.gif', '.JPG')):
            img2 = cv2.imread('images/' + a)
            kp2, des2 = sift.detectAndCompute(img2, None)
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
            matches_mask = mask.ravel().tolist()
            sum, count = 0, 0
            for match in matches:
                if count >= 10:
                    break
                sum += match.distance
                count += 1
            dis_images[i] = sum
            perfect_mask[i] = matches_mask
            masks[i] = matches
    return images[dis_images.argmin()]


@app.route('/api', methods=['POST'])
def index():
    collection = {'image': request.args['image'], 'extension': request.args['extension']}
    img_in_bytes = collection['image']
    saving_img_in_dir(f'searchedImage.jpg', img_in_bytes)
    # result = similarity_checker('searchedImage.jpg', images_in_folder)
    result = similarity_checker('searchedImage.jpg')
    return jsonify({'result': result})


@app.route('/api/saveImage', methods=['POST'])
def index2():
    collection = {'image': request.args['image'], 'imageName': request.args['imageName']}
    image_name = collection['imageName']
    file = open(f'images/{image_name}', 'wb')
    from_bytes_to_img = base64.b64decode(collection['image'])
    file.write(from_bytes_to_img)
    file.close()
    return jsonify({'image': collection['image'], 'imageName': collection['imageName']})


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # ---- for connecting api with ip address in mobile device
    # pip install pyopenssl for ssl_context='adhoc'
    app.run(ssl_context='adhoc', host='0.0.0.0', debug=True, port=5000)
