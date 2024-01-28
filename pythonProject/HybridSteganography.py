import cv2
import numpy as np


def text_to_binary(message):
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    return binary_message


def binary_to_text(binary_message):
    text_message = ''.join(chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8))
    return text_message


def lsb_encode(cover_image, message):

    binary_message = text_to_binary(message)

    flat_cover = cover_image.flatten()
    if len(binary_message) > len(flat_cover):
        raise ValueError("Binary message is too large for the cover image")

    for i in range(len(binary_message)):
        flat_cover[i] = (flat_cover[i] & 0b11111110) | int(binary_message[i])

    stego_image = flat_cover.reshape(cover_image.shape)

    return stego_image.astype(np.uint8)


def phase_encode(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    phase_message = np.random.random(gray_image.shape) * 2 * np.pi

    f_transform = np.fft.fft2(gray_image)

    magnitude = np.abs(f_transform)
    phase = np.angle(f_transform)

    encoded_phase = phase + phase_message

    encoded_transform = magnitude * np.exp(1j * encoded_phase)

    encoded_image = np.fft.ifft2(encoded_transform).real
    encoded_image = cv2.normalize(encoded_image, None, 0, 255, cv2.NORM_MINMAX)

    return encoded_image.astype(np.uint8)


def decode_message(stego_image, message_length):

    flat_stego = stego_image.flatten()

    binary_message = ''.join(str(pixel & 1) for pixel in flat_stego)[:message_length]

    decoded_message = binary_to_text(binary_message)

    return decoded_message



image_path = 'toji.jpg'
original_image = cv2.imread(image_path)

message = "I hate python!"

lsb_encoded_image = lsb_encode(original_image.copy(), message)

hybrid_encoded_image = phase_encode(lsb_encoded_image.copy())

message_length = len(text_to_binary(message))
lsb_decoded_message = decode_message(lsb_encoded_image, message_length)
hybrid_decoded_message = decode_message(hybrid_encoded_image, message_length)
print("Original Message:", message)
print("LSB Decoded Message:", lsb_decoded_message)
print("Hybrid Decoded Message:", lsb_decoded_message)

cv2.imshow('Original Image', original_image)
cv2.imshow('LSB Encoded Image', lsb_encoded_image)
cv2.imshow('Hybrid Encoded Image', hybrid_encoded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
