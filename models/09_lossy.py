import cv2

def compress_image(img, quality = 10):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)

    return cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
