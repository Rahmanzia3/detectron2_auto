


import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import random
import os
import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
from detectron2.structures import BoxMode
import itertools
from detectron2.engine import DefaultTrainer
from argparse import ArgumentParser
import requests
from tqdm import tqdm
import subprocess
from zipfile import ZipFile 

import wget
import os
'''
DATA_Folder
  ├── test
  ├── test_labels.csv
  ├── train
  └── train_labels.csv
      names.txt

classes = ['Raspberry_Pi_3', 'Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora']


'''
def get_microcontroller_dicts(csv_file, img_dir,classes):

      df = pd.read_csv(csv_file)
      df['filename'] = df['filename'].map(lambda x: img_dir+x)

      df['class_int'] = df['class'].map(lambda x: classes.index(x))

      dataset_dicts = []
      for filename in df['filename'].unique().tolist():
          record = {}
          
          height, width = cv2.imread(filename).shape[:2]
          
          record["file_name"] = filename
          record["height"] = height
          record["width"] = width

          objs = []
          for index, row in df[(df['filename']==filename)].iterrows():
            obj= {
                'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': row['class_int'],
                "iscrowd": 0
            }
            objs.append(obj)
          record["annotations"] = objs
          dataset_dicts.append(record)

      return dataset_dicts


def find_data_folder(source_path):
    sub_folders = os.listdir(source_path)
    for x in sub_folders:
        full_path = os.path.join(source_path,x)
        check_folde = os.path.isdir(full_path)
        if check_folde is True:
            # data fodler
            check_desired_folder = os.listdir(full_path)
            if len(check_desired_folder) >= 2:
                data_folder = full_path

                # print(data_folder)
    return data_folder


def un_zip(source,destination):
        file_name = source
      
        # opening the zip file in READ mode 
        with ZipFile(file_name, 'r') as zip: 
        # printing all the contents of the zip file 
            zip.printdir() 
          
            # extracting all the files 
            print('Extracting all the files now...') 

            # ######## ADD DESTINATION LOCATION HERE
            zip.extractall(destination) 
            print('Done!') 


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination) 




if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_path", default= '/home/tericsoft/team_alpha/all_networks/detectron2/zeta', help="Where all custom data is stored")
    parser.add_argument("--batch", default=4, help="Batch size")
    parser.add_argument("--image_type", default= 'jpg,png,jpge' , help="Image extentions(split by comma)")
    parser.add_argument("--iterations", default=1000, help="Number of classes")
    parser.add_argument("--cfg_model", default='COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml', help="CFG model")
    parser.add_argument("--download_url", default='1RslmRUjzYLxgpPyzSs1BP3ggCXGJRgdj', help="Download link")
    # parser.add_argument("--download_url", default='/home/tericsoft/team_alpha/all_networks/detectron2/mask/zeta', help="Download link")
    
    
    parser.add_argument("--project", default='mask', help="Give project name")

    opt = parser.parse_args()

    current_dir = os.getcwd()

    project_dir = os.path.join(current_dir,opt.project)

    os.makedirs(project_dir,exist_ok=True)

    google_drive_id = opt.download_url


    is_googleid_dir = os.path.isdir(google_drive_id)
    print(is_googleid_dir)

    if is_googleid_dir is True:

        data_dir = google_drive_id




    elif is_googleid_dir is False:

        is_google_id_url = google_drive_id.find('https')
        print(' checking url or not  ',is_google_id_url)
        if is_google_id_url is 0:
            print("Download from a link")
            # ENABLE THIS TO DOWNLOAD FROM LINK
            wget.download(google_drive_id,project_dir)
            list_sub_folder = os.listdir(project_dir)
            for x in list_sub_folder:
                check = x.find('.zip')
                if check != -1:
                    source_zip = os.path.join(project_dir,x)
                    destination_zip = project_dir

                    un_zip(source_zip,destination_zip)

        elif is_google_id_url == -1 and is_googleid_dir == False :

            print('Download from google drive')
            project_dir_zip = os.path.join(project_dir, 'custom_data.zip')
            print('assssssssssssssssssssssssssssssssssssssssss',project_dir_zip)
            download_file_from_google_drive(google_drive_id,project_dir_zip)
            sub_folder = os.listdir(project_dir)
            print('FFFFFFFFFFFFFFFFFFFFFFFFf')
            print(sub_folder)

            # index = sub_folder.index('.zip')
            # zip_folder = os.path.join(project_dir,sub_folder[index])
            
            for x in range(len(sub_folder)):

                is_zip = sub_folder[x].find('.zip')
                if is_zip != -1:
                    zip_folder = os.path.join(project_dir,sub_folder[x])

            un_zip(zip_folder,project_dir)

        data_dir = find_data_folder(project_dir)













    data_dir = opt.data_path
    # classes = opt.classes.split(',')
    array = opt.image_type.split(',')
    iteration_count = opt.iterations
    images_batch = opt.batch
    zoo_cfg_model = opt.cfg_model

    sub_folder = os.listdir(data_dir)

    for x in sub_folder:
      if x == 'train':
        train_diro = os.path.join(data_dir,x)
      if x == 'test':
        test_diro = os.path.join(data_dir,x)

      if x == 'test_labels.csv':
        path_test_csv = os.path.join(data_dir,x)
      if x == 'train_labels.csv':
        path_train_csv = os.path.join(data_dir,x)
      if x == 'names.txt':
        name_file = os.path.join(data_dir,x)


    file_read = open(name_file,'r')

    lines = file_read.readlines()
    classes = []
    for x in lines:
      x= x.split('\n', 1)[0]
      # print(x)
      # print(type(x))

      classes.append(x)



    df = pd.read_csv(path_train_csv)
    df.head()
    # Reading frm csv

    for d in [ "test" , "train"]:

      DatasetCatalog.register(data_dir+'/'+ d, lambda d=d: get_microcontroller_dicts(data_dir+'/'+ d + '_labels.csv', data_dir+'/'+ d+'/',classes))
      MetadataCatalog.get(data_dir+'/' + d).set(thing_classes=classes)
    microcontroller_metadata = MetadataCatalog.get(train_diro)


    #  Training configurations


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_cfg_model))
    cfg.DATASETS.TRAIN = (train_diro,)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_cfg_model)
    cfg.SOLVER.IMS_PER_BATCH = images_batch
    cfg.SOLVER.MAX_ITER = iteration_count
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(len(classes))

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    # ________________________________________________________________
    #           STRTS TRAINING
    # ----------------------------------------------------------------
    trainer = DefaultTrainer(cfg) 



    print(trainer)
    trainer.resume_or_load(resume=False)
    trainer.train()






# https://drive.google.com/u/0/uc?id=1RslmRUjzYLxgpPyzSs1BP3ggCXGJRgdj&export=download














































    # ________________________________________________________________
    # ----------------------------------------------------------------
    """## Use model for inference

    Now, we can perform inference on our validation set by creating a predictor object.
    """

    # cfg.MODEL.WEIGHTS = os.path.join(weights_path)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
    # cfg.DATASETS.TEST = (test_diro, )
    # predictor = DefaultPredictor(cfg)

    # sub_path = os.listdir(test_diro)
    # images_path = []
    # for x in sub_path:
    #     extention = x.rpartition('.')[-1]
    #     # print(extention[-1])
    #     if extention in array:
    #         full_path = os.path.join(test_diro,x)
    #         images_path.append(full_path)




    # for d in images_path:    

    #     im = cv2.imread(d)
    #     outputs = predictor(im)
    #     v = Visualizer(im[:, :, ::-1], metadata=microcontroller_metadata, scale=0.8)
    #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     plt.figure(figsize = (14, 10))
    #     plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    #     plt.show()



