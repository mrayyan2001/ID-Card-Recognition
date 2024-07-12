import PIL.Image
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pytesseract import pytesseract, Output
import pandas as pd

fields = {"ID Number": None, "Name": None, "Nationality": None}


pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
myconfiq = r"--psm 11 --oem 3"


def remove_background(self):
    contours, _ = cv2.findContours(
        self.th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # the contours are stored in a list(each contour represet as array)
    cv2.drawContours(self.id_card, contours, -1, (0, 255, 0), 2)

    largest_contour = max(
        contours, key=cv2.contourArea
    )  # save just the area that have large contours
    mask = np.zeros_like(self.gray_id)
    cv2.drawContours(mask, [largest_contour], -1, (255), -1)

    background_removed = cv2.bitwise_and(self.id_card, self.id_card, mask=mask)


# Upload Image
img = st.file_uploader("Image", ["png", "jpeg", "jpg"])

if img is not None:
    # Convert img to numpy array
    img = np.array(Image.open(img))

    # Show Image
    st.image(img)

    # Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(grayscale)

    # Threshold
    _, thr = cv2.threshold(grayscale, 170, 255, cv2.THRESH_BINARY)
    st.image(thr)

    data = pd.DataFrame(
        pytesseract.image_to_data(
            grayscale, config=myconfiq, output_type=Output.DATAFRAME
        )
    )
    data = data[data["conf"] > 65]
    data["group"] = pd.cut(
        data["top"], range(data["top"].min() - 10, data["top"].max() + 10, 10)
    )

    text = " ".join(data["text"])
    indices = []
    for i in fields:
        try:
            indices.append(text.index(i))
        except:
            pass

    st.write(fields)

    for i, f in enumerate(fields):
        st.write(f)
        start = indices[i] + len(f) + 1
        try:
            end = indices[i + 1]
        except:
            end = -1
        st.write("- start index: ", start)
        st.write("- end index: ", end)
        fields[f] = text[start:end]
        st.write(text[start:end])

    st.write(fields)

    boxes = len(data)
    new_img = img.copy()
    for i in range(boxes):
        (x, y, width, height) = (
            data["left"].iloc[i],
            data["top"].iloc[i],
            data["width"].iloc[i],
            data["height"].iloc[i],
        )
        new_img = cv2.rectangle(
            new_img,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
    st.image(new_img)
