import os
import pymupdf
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import time

# Detect faces using haarcascade
def detect_faces(img):
    detec = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detec.detectMultiScale(gray, 1.3, 3)

    return faces

# Detect smiles using haarcascade
def detect_smiles(img):
    smile_cascade = cv2.CascadeClassifier("./haarcascade_smile.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 10)

    return smiles

def detect_faces_in_file(file):
    FACES_LIMIT = 50
    year = file.split('.')[0] # The files have their names like <year>.pdf

    pdf_path = os.path.join("./yearbooks", file)
    print("Getting faces from", pdf_path)

    count_faces = 0
    
    # Iterate for each page
    for page in pymupdf.open(pdf_path):
        # Get page as image
        pix = page.get_pixmap()

        # Tranfsorm image to np.arraay 
        img =  np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        
        # Detect
        faces = detect_faces(img)

        # Create PNG of each detected face
        for i, (x, y, w, h) in enumerate(faces):
            crop = img[y:y+h,x:x+w]
            cv2.imwrite(f"./images/{year}/{page.number}_{i}.png", crop)
            count_faces += 1

            # If reached the limit of faces to detect, stops
            # reading this file
            if count_faces >= FACES_LIMIT:
                break
 
        # If reached the limit of faces to detect, stop s
        # reading this file       
        if count_faces >= FACES_LIMIT:
            break

# For each year, count the number of faces that are smiling
def detect_smiles_at_year(year):
    files = os.listdir(f"./images/{year}")
    count_smiles = 0

    # For each face, detect smile.
    for file in files:
        img = cv2.imread(f"./images/{year}/{file}")
        smiles = detect_smiles(img)
        if len(smiles) > 0:
            count_smiles += 1

    smile = {
        "year": year,
        "smile_count": count_smiles,
        "nonsmile_count": len(files) - count_smiles,
        "smile_factor": count_smiles / len(files) 
    }
    
    return pd.DataFrame([smile])

if __name__ == '__main__':
    years = []
    
    nprocs = 4
    files = os.listdir("./yearbooks")

    # Get years and create folders for each year
    for file in files:
        year = file.split(".")[0]
        years.append(year)
        
        images_path = f"./images/{year}/"
        
        if not os.path.isdir(images_path):
            os.makedirs(images_path)

    print("Starting face detection...")
    
    # Create PNG file for each detected face
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = executor.map(
            detect_faces_in_file,
            files,
        )

    print("Got all faces. Analysing smiles...")

    df_smiles = pd.DataFrame()

    # Create DataFrame containing the smiles anaysis
    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = [executor.submit(detect_smiles_at_year,year) for year in years]

        wait(futures)

        for result in futures:
            df_smiles = pd.concat([df_smiles, result.result()], ignore_index=True)

    print("All smiles analysed. Creating CSV..")

    # Sort by year
    df_smiles = df_smiles.sort_values(by=['year'], ignore_index=True)
    df_smiles.to_csv("smile_analysis.csv", index=False)

    print("All done.")

