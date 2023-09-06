import cv2
import matplotlib.pyplot as plt 
import os
import configparser as cf
import sys
from train_model import upload_file_to_s3, get_dataset , update_not_found_image

try: 
    global ENV
    ENV = str(sys.argv[1])
    if ENV == 'prd':
        setup = cf.ConfigParser()
        setup.read("config_prd.ini")
    elif ENV == 'dev':
        setup = cf.ConfigParser()
        setup.read("config_dev.ini")
    else: 
        print("Environment incorrect ...")
        exit()
except Exception as error:
    raise error

BUCKET = setup.get("AWS_S3","BUCKET")
cur_dir = os.getcwd()

def upload_file():
    rows = get_dataset()
    rawImgs = []
    labels = []
    count = 0
    for idx , data in enumerate(rows):
        aws_path = data[2].split("/")[:-1]
        image_name = data[9]
        label = data[1]
        try: 
            img = cv2.imread(cur_dir+'/TrainingImages/'+image_name , cv2.COLOR_BGR2RGB)
            img = cv2.resize(img ,(255,255))
            if label == 1:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Normal/' + image_name)
            elif label == 2:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Diabetes/' + image_name)
            elif label == 3:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Glaucoma/' + image_name)
            elif label == 4:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Age related Macular Degeneration/' + image_name)
            elif label == 5:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Hypertension/' + image_name)
            elif label == 6:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Pathological Myopia/' + image_name)
            elif label == 7:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Other diseases or abnormalities/' + image_name)
            elif label == 8:
                upload_file_to_s3(BUCKET,cur_dir+'/TrainingImages/'+image_name,'dataset/Cataract/' + image_name)
            print(idx, "Upload file to S3 success from ",cur_dir+'/TrainingImages/'+image_name,"to",'dataset/Normal/' + image_name)
        except:
            update_not_found_image(data[0])
            print("Upload file failed")
            continue
    return rawImgs,labels


if __name__ == '__main__':
    upload_file()