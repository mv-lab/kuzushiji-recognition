import numpy as np
from sklearn.model_selection import GroupKFold
import cv2
from tqdm import tqdm_notebook as tqdm

from tom_functions.utils import *


def get_img_size(idx, data_df):
    img_name = data_df.loc[idx, 'image_id'] + '.jpg'
    path = expand_path_img(img_name)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,c = img.shape
    aspect_ratio = h/w
    return h,w, aspect_ratio


def get_train_valid(df_train, fold):
    df_train['book_title'] = df_train['image_id'].apply(lambda x:x.split('_')[0])
    df_train['book_title'] = df_train['book_title'].apply(lambda x:x.split('-')[0])

    #train aspect_ratio
    h_list = []
    w_list = []
    aspect_ratio_list = []
    for idx in tqdm(range(len(df_train))):
        h,w,aspect_ratio = get_img_size(idx, df_train)
        h_list.append(h)
        w_list.append(w)
        aspect_ratio_list.append(aspect_ratio)

    #idxs satisfying aspect ratio criteria
    validation_idx = []
    for idx, x in enumerate(tqdm(aspect_ratio_list)): 
        if x>= 1.25 and x<=1.75:
            validation_idx.append(idx)
        else:
            pass
    validation_idx = np.array(validation_idx)
    print('len(validation_idx) / len(df_train) = ', len(validation_idx) / len(df_train))

    
    df_train_2 = df_train.iloc[validation_idx].reset_index(drop=True)
    
#     #check the book title of outliers
#     outlier_idx = np.array(list(set(df_train.index)-set(validation_idx)))
#     book_title_outlier = df_train.iloc[outlier_idx].book_title.unique()
#     book_title_validation = df_train_2.book_title.unique()
#     print('the book title of intersection = ', list(set(book_title_validation) & set(book_title_outlier)))


    #GroupKFold by book title
    book_title_group = df_train_2.book_title.values
    kfold = GroupKFold(n_splits=5)
    train_idxs = df_train_2.index
    val_idxs_list   = []
    for f, (trn,val) in enumerate(kfold.split(train_idxs,train_idxs, book_title_group)):
        val_idxs_list.append(val)


    print('len(val_idxs_list[fold]) = ', len(val_idxs_list[fold]))

    valid_df = df_train_2.iloc[val_idxs_list[fold]].reset_index(drop=True)
    trn_idxs = np.array(list(set(df_train_2.index) - set(val_idxs_list[fold])))
    train_df = df_train_2.iloc[trn_idxs].reset_index(drop=True)

    return train_df, valid_df