import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import numpy as np
# source and credits:
# https://raw.githubusercontent.com/datitran/raccoon_dataset/master/xml_to_csv.py

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        #print(xml_file)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            try:
		    value = (root.find('filename').text,
		             int(root.find('size')[0].text),
		             int(root.find('size')[1].text),
		             member[0].text,
		             int(member[4][0].text),
		             int(member[4][1].text),
		             int(member[4][2].text),
		             int(member[4][3].text)
		             )
            except:
                image_name=(root.find('filename').text.split('.')[0]+".jpg")
                w,h,c = (np.shape(cv2.imread(path+"/"+image_name)))
                #print(member[2][0].text)
                #print(root.find('object').text)     
		value = (root.find('filename').text,
		             int(w),
		             int(h),
		             member[0].text,
		             int(member[2][0].text),
		             int(member[2][1].text),
		             int(member[2][2].text),
		             int(member[2][3].text)
		             )
               
            xml_list.append(value)
	    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def train():
    image_path = os.path.join('/home/dl/SurajDL/MaskNonMaskProject/Codes/darknet/Version_3/Dataset/train/')
    xml_df = xml_to_csv(image_path)
    labels_path = os.path.join(os.getcwd(), 'train.csv')
    xml_df.to_csv(labels_path, index=None)
    print('> tf_wider_train - Successfully converted xml to csv.')

def val():
    image_path = os.path.join('/home/dl/SurajDL/MaskNonMaskProject/Codes/darknet/Version_3/Dataset/val')
    xml_df = xml_to_csv(image_path)
    labels_path = os.path.join(os.getcwd(), 'val.csv')
    xml_df.to_csv(labels_path, index=None)
    print('> tf_wider_val -  Successfully converted xml to csv.')

train()
val()
