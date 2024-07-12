import cv2
import numpy as np
import pandas as pd
import streamlit as st
from pytesseract import pytesseract, Output
from PIL import Image

pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
myconfiq = r"--psm 11 --oem 3"


def remove_background(img: np.ndarray):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thr = cv2.threshold(grayscale, 170, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # the contours are stored in a list(each contour represet as array)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    largest_contour = max(
        contours, key=cv2.contourArea
    )  # save just the area that have large contours
    mask = np.zeros_like(grayscale)
    cv2.drawContours(mask, [largest_contour], -1, (255), -1)

    background_removed = cv2.bitwise_and(img, img, mask=mask)
    return background_removed


def extract_info(img: np.ndarray):
    img = remove_background(img)
    fields = {"ID Number": None, "Name": None, "Nationality": None}

    # Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    data = pd.DataFrame(
        pytesseract.image_to_data(
            grayscale, config=myconfiq, output_type=Output.DATAFRAME
        )
    )

    data = data[data["conf"] > 70]

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

    text = " ".join(data["text"])
    indices = []
    for i in fields:
        try:
            indices.append(text.index(i))
        except:
            pass

    for i, f in enumerate(fields):
        try:
            start = indices[i] + len(f) + 1
        except:
            break
        try:
            end = indices[i + 1]
        except:
            end = -1
        fields[f] = text[start:end]

    return fields


def update(img: np.ndarray):
    data = extract_info(img)
    st.session_state.id = str(data["ID Number"]).strip("/").strip(" ")
    st.session_state.name = data["Name"]
    st.session_state.nationality = data["Nationality"]


def clear():
    st.session_state.name = None
    st.session_state.id = None
    st.session_state.nationality = None


def main():
    if "name" not in st.session_state:
        st.session_state.name = None
    if "id" not in st.session_state:
        st.session_state.id = None
    if "gender" not in st.session_state:
        st.session_state.nationality = None
    if "img" not in st.session_state:
        st.session_state.img = None

    img = st.file_uploader(
        "ID Card",
        ["png", "jpg", "jpeg"],
    )

    if img is not None:
        # Convert img to numpy array
        img = np.array(Image.open(img))
        st.image(img)

    if img is None:
        clear()
    else:
        st.session_state.img = img
        update(img)

    col1, col2, col3 = st.columns(3)
    with col1:
        id = st.text_input(label="ID Number", value=st.session_state.id, key="id_input")

    with col2:
        name = st.text_input(
            label="Name", value=st.session_state.name, key="name_input"
        )

    with col3:
        nationality = st.text_input(
            label="Nationality", value=st.session_state.nationality, key="gender_input"
        )

    submit = st.button("Submit", use_container_width=True)

    if submit == True:
        if id is None:
            st.toast("Can't add None", icon="⚠️")
        elif id in st.session_state.df["ID Number"].unique():
            st.toast("Already Added", icon="⚠️")
        else:
            st.session_state.df = pd.concat(
                [
                    pd.DataFrame(
                        [
                            [
                                id,
                                name,
                                nationality,
                            ],
                        ],
                        columns=st.session_state.df.columns,
                    ),
                    st.session_state.df,
                ],
                ignore_index=True,
            )
            st.session_state.df.reindex()

    selected = st.dataframe(
        st.session_state.df,
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    try:
        st.session_state.df = pd.read_csv("./data.csv", dtype=str)
    except Exception as ex:
        if "df" not in st.session_state:
            st.session_state.df = pd.DataFrame(
                columns=["ID Number", "Name", "Nationality"]
            )
    main()
    if len(st.session_state.df) != 0:
        st.session_state.df.to_csv("./data.csv", index_label=False)
