# required library

import re
import cv2
import pytesseract
import matplotlib.pyplot as plt
from ultralytics import YOLO
import streamlit as st

#OCR and YOLO model
model = YOLO('best.pt')
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\tesseract.exe'

vehicle_data = {
    "HR26DK8337": {"name": "Rahul Kumar", "mobile": "9876543210"},
    "KL01CD5678": {"name": "Anil Das", "mobile": "9123456780"},
    "KL05EF9012": {"name": "Priya Nair", "mobile": "9988776655"},
    "KL08GH3456": {"name": "Arjun Menon", "mobile": "9871234567"}
}

def find_x(path):

    result = model(path)

    for res in result:
        for box in res.boxes.xyxy.tolist():
            x1, y1, x2, y2 = box 

    # convert coordinates to integers once after the loop
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    img1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    img_in_frame = img1[y1:y2, x1:x2]
    img_in_frame = cv2.cvtColor(img_in_frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite("saved_image.jpg", img_in_frame)


    data = pytesseract.image_to_data(img_in_frame, output_type=pytesseract.Output.DICT)
    for i, text in enumerate(data['text']):
        
        if data['conf'][i] > 50 and re.search(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', text):
            number_plate = text
        else:
            number_plate = "unkown"
   

    if number_plate in vehicle_data:
        
        Person_Name = vehicle_data[number_plate]["name"]
        
        Mobile_Number=vehicle_data[number_plate]["mobile"]        
        
    else:
        Person_Name = "Unknown"
        
        Mobile_Number="Unknown" 

    return number_plate,Person_Name,Mobile_Number
    
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

find_x('Swift.jpg')


import streamlit as st
from PIL import Image
import os



st.title("Vehicle Number Plate Detection")

uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    input_image_path = "uploaded_image.jpg"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

   
    

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(input_image_path, use_container_width=True)

    with col2:
        st.subheader("Detected Number Plate")
        if os.path.exists("saved_image.jpg"):
            st.image("saved_image.jpg", use_container_width=True)
        else:
            st.warning("Number plate image not found")

    number_plate,Person_Name,Mobile_Number = find_x(input_image_path)
    # Display vehicle number

    st.markdown("### Vehicle Details")
    st.success(f"Vehicle Number: {number_plate}")
    st.write(f"Person Name: {Person_Name}")
    st.write(f"Mobile Number: {Mobile_Number}")