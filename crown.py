import cv2
import numpy as np

# Load the mask image
mask = cv2.imread('icons/mask.png', cv2.IMREAD_UNCHANGED)
crown = cv2.imread('icons/crown.png', cv2.IMREAD_UNCHANGED)

# Initialize the webcam
# cap = cv2.VideoCapture(0)

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Add an alpha channel if the image doesn't have one
def reshape_image(image):
    if image.shape[2] != 4:
        b, g, r = cv2.split(image)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Create a fully opaque alpha channel
        image = cv2.merge((b, g, r, alpha))
    return image


crown = reshape_image(crown)
mask = reshape_image(mask)


# Add an alpha channel if the image doesn't have one
def ensure_alpha_channel(image):
    if image.shape[2] != 4:
        b, g, r = cv2.split(image)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Fully opaque alpha channel
        return cv2.merge((b, g, r, alpha))
    return image

crown = ensure_alpha_channel(crown)
mask = ensure_alpha_channel(mask)


# Overlay mask on the face
def overlay_mask(frame, mask, x, y, w, h):
    # Resize mask to fit the detected face
    mask_resized = cv2.resize(mask, (w, int(h * 0.8)))  # Adjust height as needed
    y_offset = y + int(h * 0.1)  # Align with the nose
    x_offset = x

    # Ensure ROI is within the frame boundaries
    y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + mask_resized.shape[0])
    x1, x2 = max(0, x_offset), min(frame.shape[1], x_offset + mask_resized.shape[1])

    roi = frame[y1:y2, x1:x2]
    mask_resized = mask_resized[: y2 - y1, : x2 - x1]

    # Separate the color and alpha channels
    mask_rgb = mask_resized[:, :, :3]
    mask_alpha = mask_resized[:, :, 3] / 255.0

    # Overlay the mask on the ROI
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask_alpha) + mask_rgb[:, :, c] * mask_alpha

    # Place the modified ROI back into the frame
    frame[y1:y2, x1:x2] = roi



def overlay_crown(frame, crown, x, y, w, h):
    # Resize the crown image to fit the width of the detected face
    crown_resized = cv2.resize(crown, (w, int(w * crown.shape[0] / crown.shape[1])))

    # Calculate position to overlay the crown
    y_offset = y - crown_resized.shape[0] + 10  # Adjust as needed

    # Ensure the y_offset is within the frame
    if y_offset < 0:
        y_offset = 0

    # Extract the region of interest (ROI) from the frame
    roi = frame[y_offset:y_offset + crown_resized.shape[0], x:x + w]

    # Separate the color and alpha channels of the crown image
    crown_rgb = crown_resized[:, :, :3]
    crown_alpha = crown_resized[:, :, 3] / 255.0

    # Overlay the crown on the ROI
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - crown_alpha) + crown_rgb[:, :, c] * crown_alpha

    # Place the modified ROI back into the frame
    frame[y_offset:y_offset + crown_resized.shape[0], x:x + w] = roi


def show_clown(frame, left_is_clown , right_is_clown, width):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_center = x + w // 2  # Calculate the center of the detected face

        if face_center < width // 2:  # Face is on the left side
            if left_is_clown:
                overlay_mask(frame, mask, x, y, w, h)  # Apply clown mask
        else:  # Face is on the right side
            if right_is_clown:
                overlay_mask(frame, mask, x, y, w, h)  # Apply clown mask

    return frame


def show_crown(frame , winner , width):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_center = x + w // 2  
        if face_center < width // 2:  
            if winner == 'left':
                overlay_crown(frame, crown, x, y, w, h)  
        else:  
            if winner == 'right':
                overlay_crown(frame, crown, x, y, w, h)  

    return frame