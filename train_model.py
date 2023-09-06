from pathlib import Path
from datetime import date , timedelta
from keras.models import Sequential
from keras.layers import Dense, Flatten , Conv2D, MaxPool2D , Dropout , GaussianNoise
from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2
import numpy as np
import os
import tensorflow as tf
import configparser as cf
import time
import uuid
import sys
import pymysql
import boto3
from datetime import datetime
import joblib
import json
# from function import get_dataset , download_file_from_s3 , iecho , insert_into_model_master , upload_file_to_s3

print("Please select enviroment : dev,prd")
ENV = input("Select environment : ")
print("Please input number of image ")
IMGNUM = input("Input number data : ")
print("Please input percentage of test_data ")
TESTP = float(input("Percentage test data : "))
print("Plase input epoch num ")
EPNUM = int(input("Number of epoch : "))

if ENV == 'prd':
    setup = cf.ConfigParser()
    setup.read("config_prd.ini")
elif ENV == 'dev':
    setup = cf.ConfigParser()
    setup.read("config_dev.ini")
else: 
    print("Environment incorrect ...")
    sys.exit()

BUCKET = setup.get("AWS_S3","BUCKET")

cur_dir = os.getcwd()
today = str(date.today())
model_path = cur_dir + '/model_file/model_cnn_'+today+'.h5'

diagnostic_encoding = ['Normal',
                       'Diabetes',
                       'Glaucoma',
                       'Age related Macular Degeneration',
                       'Hypertension',
                       'Pathological Myopia ',
                       'Other diseases or abnormalities',
                       'Cataract']
def connect_database():
    try:
        conn = pymysql.connect(
        host=setup.get("DATABASE","HOSTNAME")
        ,user=setup.get("DATABASE","USER")
        ,password=setup.get("DATABASE","PASSWORD")
        ,db=setup.get("DATABASE","DB")
            )
        print('------------------------------------------')
        print("DATABASE CONNECTED")
        print('------------------------------------------')
        cur = conn.cursor() 
        return cur , conn 
    except Exception as error: 
        print('------------------------------------------')
        print("CANNOT CONNECT TO DATABASE ..")
        print('------------------------------------------')
        print(str(error))
        raise error
    
def update_not_found_image(datasetid):
    cur , conn = connect_database()
    cur.execute("UPDATE TABLE HEALTH_ME.OCULAR_DATASET SET DOWNLOAD_FILE_STATUS = 'N' WHERE DATASET_ID = "+str(datasetid))
    conn.commit()
    conn.close()

def get_dataset():
    cur , conn = connect_database()
    cur.execute("SELECT CAST(AVG(COUNT_NUM) AS UNSIGNED) FROM (SELECT COUNT(*) AS COUNT_NUM,DIAGNOSTIC_KEYWORDS_ID  FROM OCULAR_DATASET OD GROUP BY DIAGNOSTIC_KEYWORDS_ID ) AS T1")
    # get_average_keyword = cur.fetchall()
    get_average_keyword = [[IMGNUM]]
    # cur.execute("SELECT * FROM OCULAR_DATASET")
    print("Executing statement : (SELECT * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 1 LIMIT " + str(get_average_keyword[0][0]) + ")" 
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 2 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 3 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 4 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 5 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 6 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 7 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 8 LIMIT " + str(get_average_keyword[0][0]) + ");"
    )
    cur.execute("(SELECT * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 1 LIMIT " + str(get_average_keyword[0][0]) + ")" 
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 2 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 3 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 4 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 5 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 6 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 7 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 8 LIMIT " + str(get_average_keyword[0][0]) + ");"
    )
    row = cur.fetchall()
    cur.execute("SELECT COUNT(1),DIAGNOSTIC_KEYWORDS_ID FROM ( " + 
                "(SELECT * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 1 LIMIT " + str(get_average_keyword[0][0]) + ")" 
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 2 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 3 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 4 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 5 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 6 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 7 LIMIT " + str(get_average_keyword[0][0]) + ")"
                " UNION ALL (SELECT  * FROM OCULAR_DATASET WHERE DIAGNOSTIC_KEYWORDS_ID = 8 LIMIT " + str(get_average_keyword[0][0]) + ")) AS T1 GROUP BY DIAGNOSTIC_KEYWORDS_ID;"
    )
    group_by = cur.fetchall()
    print(group_by)
    conn.close()
    return row

def insert_into_model_master(MODEL_ID,MODEL_FILENAME,ACCURACY_PERCENTAGE,ELAPSED_TIME,ACTIVE_FLAG,CREATE_DATE,CREATE_USER,UPDATE_DATE,UPDATE_USER,REMARK):
    cur , conn = connect_database()
    cur.execute(
        """
        INSERT INTO MODEL_MASTER ( 
            MODEL_ID,MODEL_FILENAME,ACCURACY_PERCENTAGE,ELAPSED_TIME,ACTIVE_FLAG,CREATE_DATE,CREATE_USER,UPDATE_DATE,UPDATE_USER,REMARK )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
    (
        MODEL_ID,MODEL_FILENAME,ACCURACY_PERCENTAGE,ELAPSED_TIME,ACTIVE_FLAG,CREATE_DATE,CREATE_USER,UPDATE_DATE,UPDATE_USER,REMARK
    ))    
    conn.commit()
    cur.execute(
    "UPDATE MODEL_MASTER SET ACTIVE_FLAG = 'N' WHERE MODEL_ID <> "+ str(MODEL_ID)
    )
    conn.commit()
    conn.close()

def download_file_from_s3(bucket,filename,destination):
    KEY_ID = setup.get("AWS_KEY","KEY_ID")
    SECRET_KEY = setup.get("AWS_KEY","SECRET_KEY")
    s3 = boto3.client('s3',aws_access_key_id=KEY_ID,aws_secret_access_key=SECRET_KEY)
    s3.download_file(bucket, filename , destination)

def upload_file_to_s3(bucket,filename,destination):
    KEY_ID = setup.get("AWS_KEY","KEY_ID")
    SECRET_KEY = setup.get("AWS_KEY","SECRET_KEY")   
    s3 = boto3.client('s3',aws_access_key_id=KEY_ID,aws_secret_access_key=SECRET_KEY)
    s3.upload_file(filename, bucket , destination)

def predict(input_data):
    cnn_from_pickle = joblib.load(setup.get("MODEL","MODEL_PATH"))
    cnn_from_pickle.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='categorical_crossentropy',
              metrics= ['accuracy'])
    diagnostic_keyword = [2, #'Diabetes',
        6, #'Pathological Myopia',
        4, #'Age related Macular Degeneration',
        3, #'Glaucoma',          # 
        8, #'Cataract',          
        5, #'Hypertension',      
        1, #'Normal',            
        7 #'Other diasease'     
    ]

    answer = cnn_from_pickle.predict(input_data) * 100
    answer = sorted([(key, value) for i, (key, value) in enumerate(zip(np.round(answer[0],2), diagnostic_keyword))],reverse=True)
    answer = answer[0] + answer[1] + answer[2]

    diagnostic_result_1 = json.dumps(str(answer[1])).strip('\\"')
    diagnostic_percentage_1 = json.dumps(str(answer[0])).strip('\\"')
    diagnostic_result_2 = json.dumps(str(answer[3])).strip('\\"')
    diagnostic_percentage_2 = json.dumps(str(answer[2])).strip('\\"')
    diagnostic_result_3 = json.dumps(str(answer[5])).strip('\\"')
    diagnostic_percentage_3 = json.dumps(str(answer[4])).strip('\\"')

    return diagnostic_result_1 , diagnostic_percentage_1 , diagnostic_result_2 , diagnostic_percentage_2 , diagnostic_result_3 , diagnostic_percentage_3

# def preprocessingImage(image):
#     img = cv2.imread(image)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     test_img_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)
#     cnts = cv2.findContours(test_img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#     for c in cnts:
#         x,y,w,h = cv2.boundingRect(c)
#         test_img_ROI = img[y:y+h, x:x+w]
#         break
#     return test_img_ROI

def iecho(checkpoint):
    print("")
    now = datetime.now()
    print("###############################################################")
    print("#    START STEP {:45s} #".format(checkpoint))
    print("#    TIME :",now,"                       #")         
    print("###############################################################")
    print("")
    return True

def download_all_images(bucket):
    rows = get_dataset()
    for idx , data in enumerate(rows):
        target_image_name = data[2].split("/") 
        try: 
            if data[1] == 1:
                download_file_from_s3(bucket,'dataset/Normal/'+target_image_name[-1], cur_dir + '/training_data_temp/' + target_image_name[-1])
            elif data[1] == 2:
                download_file_from_s3(bucket,'dataset/Diabetes/'+target_image_name[-1],  cur_dir + '/training_data_temp/' + target_image_name[-1])
            elif data[1] == 3:
                download_file_from_s3(bucket,'dataset/Glaucoma/'+target_image_name[-1],  cur_dir + '/training_data_temp/' + target_image_name[-1])
            elif data[1] == 4:
                download_file_from_s3(bucket,'dataset/Age related Macular Degeneration/'+target_image_name[-1],  cur_dir + '/training_data_temp/' +target_image_name[-1])
            elif data[1] == 5:
                download_file_from_s3(bucket,'dataset/Hypertension/'+target_image_name[-1],  cur_dir + '/training_data_temp/' + target_image_name[-1])
            elif data[1] == 6:
                download_file_from_s3(bucket,'dataset/Pathological Myopia/'+target_image_name[-1],  cur_dir + '/training_data_temp/' + target_image_name[-1])
            elif data[1] == 7:
                download_file_from_s3(bucket,'dataset/Other diseases or abnormalities/'+target_image_name[-1],  cur_dir + '/training_data_temp/' + target_image_name[-1])
            elif data[1] == 8:
                download_file_from_s3(bucket,'dataset/Cataract/'+target_image_name[-1],  cur_dir + '/training_data_temp/' + target_image_name[-1])
            print(idx+1," Download image success :",data[2])
        except Exception as error: 
            print(idx+1," Download image failed :",data[2])
            print(error)

def img2data():
    rows = get_dataset()
    rawImgs = []
    labels = []
    for idx , data in enumerate(rows):
        target_image_name = data[2].split("/") 
        img = cur_dir + '/training_data_temp/' + target_image_name[-1]
        exist = Path(img).is_file()
        if exist == True:
            try:
                img_data = cv2.imread(img , cv2.COLOR_BGR2GRAY)
                img_data = cv2.resize(img_data ,(255,255))
                # img_data = cv2.resize(img_data , (300,300))
                img_data = np.array(img_data, dtype='float32')
                if data[1] == 1:
                    rawImgs.append(img_data)
                    labels.append([1,0,0,0,0,0,0,0])
                elif data[1] == 2:
                    rawImgs.append(img_data)
                    labels.append([0,1,0,0,0,0,0,0])
                elif data[1] == 3:
                    rawImgs.append(img_data)
                    labels.append([0,0,1,0,0,0,0,0])
                elif data[1] == 4:
                    rawImgs.append(img_data)
                    labels.append([0,0,0,1,0,0,0,0])
                elif data[1] == 5:
                    rawImgs.append(img_data)
                    labels.append([0,0,0,0,1,0,0,0])
                elif data[1] == 6:
                    rawImgs.append(img_data)
                    labels.append([0,0,0,0,0,1,0,0])
                elif data[1] == 7:
                    rawImgs.append(img_data)
                    labels.append([0,0,0,0,0,0,1,0])
                elif data[1] == 8:
                    rawImgs.append(img_data)
                    labels.append([0,0,0,0,0,0,0,1])
                print(idx+1," Read image success :",img , "Dianostic keywords : ",data[1])
            except:
                continue
        else: 
            print(idx+1," Read image failed  :",img)
    return rawImgs,labels

def transform_data(test_data):
    a_train , b_train = img2data()
    x_train, x_test, y_train, y_test = train_test_split(a_train, b_train, test_size=test_data, random_state=42)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print("###############################################################")
    print("Count number of train label : ", len(x_train))
    print("Count number of train feature : ", len(y_train))
    print("Shape of label : ",x_train.shape)
    print("Shape of feature : ",y_train.shape)
    print("Count number of test label : ", len(x_test))
    print("Count number of test feature : ", len(y_test))
    print("Shape of label : ",x_test.shape)
    print("Shape of feature : ",y_test.shape)
    print("###############################################################")
    return x_train, x_test, y_train, y_test

def train_model(destination_path,feature_train,label_train,feature_test,label_test,epoch):
    try:
        model0 = Sequential([
                Conv2D(255, (3,3), activation='relu', input_shape=(255, 255, 3)),
                MaxPool2D(2),
                Conv2D(255,(3,3) , activation='relu'),
                MaxPool2D(pool_size=(2,2)),
                Dense(16),
                Flatten(),
                Dense(8, activation='softmax') #softmax for one hot . . # sigmoid for 0/1
            ])
        model0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss='categorical_crossentropy', 
                    metrics= ['accuracy'])
        model0.fit(feature_train,label_train ,batch_size=32,epochs=epoch ,validation_data=(feature_test, label_test))
        model0.save(destination_path)
        upload_file_to_s3('health-me',destination_path,'model/model_cnn_'+today+'.h5')
        print("Evaluate :",model0.evaluate(feature_test))
        return model0.evaluate(feature_train,label_train)
    except Exception as Error:
        print("Failed to train model")
        print(Error)

if __name__ == '__main__':
    try:
        checkpoint = 0
        # check file exist 

        os.system("mkdir "+ cur_dir + '/training_data_temp/')
        iecho("Download File")
        start = time.time()
        checkpoint = 1
        download_all_images(BUCKET)
        end = time.time()
        print("")
        print("###############################################################")
        print("#   Elapsed time : {:42} #".format(end-start))
        print("###############################################################")
        print("")
        checkpoint = 2 
        x_train, x_test, y_train, y_test = transform_data(TESTP)
        iecho("Train model")
        start = time.time()
        checkpoint = 3
        accuracy = train_model(destination_path=model_path
                               ,feature_train=x_train
                               ,label_train=y_train
                               ,feature_test=x_test
                               ,label_test=y_test
                               ,epoch=EPNUM
                               )
        end = time.time()
        elapsed_time = end-start
        td = timedelta(seconds=elapsed_time)
        checkpoint = 4 
        id = int(uuid.uuid4().int & (1<<64)-2)
        insert_into_model_master(id,model_path,accuracy[1] * 100 ,td,"Y",date.today(),None,None,None,None)
        print("")
        print("###############################################################")
        print("#   Elapsed time : {:42} #".format(end-start))
        print("###############################################################")
        print("")
    except Exception as error: 
        print("")
        print("###############################################################")
        print("#   Error step   : ",checkpoint)
        print(error)
        print("###############################################################")
        print("")
    # finally:
        # os.system("rm "+ cur_dir + '/training_data_temp/*.jpg')