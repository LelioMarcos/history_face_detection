import os
import pymupdf
import cv2
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import time

def detect_faces(img):
    detec = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detec.detectMultiScale(gray, 1.3, 3)

    return faces

def detect_smiles(img):
    smile_cascade = cv2.CascadeClassifier("./haarcascade_smile.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 10)

    return smiles

def detect_faces_in_file(file):
    FACES_LIMIT = 30 
    year = file.split('.')[0]

    pdf_path = os.path.join("./yearbooks", file)
    print("Getting faces from", pdf_path)
    # Create path to store the image of each face
    count_faces = 0

    pdf = pymupdf.open(pdf_path)

    for page in pdf:
        pix = page.get_pixmap()
        img =  np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        faces = detect_faces(img)

        for i, (x, y, w, h) in enumerate(faces):
            crop = img[y:y+h,x:x+w]
            cv2.imwrite(f"./images/{year}/{page.number}_{i}.png", crop)
            count_faces += 1

            if count_faces >= FACES_LIMIT:
                break
        
        if count_faces >= FACES_LIMIT:
            break

def detect_smiles_at_year(year):
    files = os.listdir(f"./images/{year}")
    count_smiles = 0

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

    start = time.time()    
    for file in files:
        year = file.split(".")[0]
        years.append(year)
        
        images_path = f"./images/{year}/"
        
        if not os.path.isdir(images_path):
            os.makedirs(images_path)

    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = executor.map(
            detect_faces_in_file,
            files,
        )

    print("Got all faces. Analysing smiles...")

    df_smiles = pd.DataFrame()

    with ProcessPoolExecutor(max_workers=nprocs) as executor:
        futures = [executor.submit(detect_smiles_at_year,year) for year in years]

        wait(futures)

        for result in futures:
            df_smiles = pd.concat([df_smiles, result.result()], ignore_index=True)

    df_smiles = df_smiles.sort_values(by=['year'], ignore_index=True)

    df_smiles.to_csv("smile_analysis.csv", index=False)

