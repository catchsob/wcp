#!/usr/bin/env python
# coding: utf-8

# # Taiwan License Plate Generator

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import choices, randint
from glob import glob


# In[96]:


class TaiwanLicensePlateGenerator:
    # New SPEC
    CAR_PLATE_W = 380 #380mm 車牌寬
    CAR_PLATE_H = 160 #160mm 車牌高
    CAR_PLATE_P = int(CAR_PLATE_W/2)-CAR_PLATE_H #湊成 2:1 (w:h) 需要的 padding
    CAR_CHAR_AMT = 7 #字元數 (不含 '-'；'-' 另計)
    CAR_SEP_POS = 3 #count from 0 ('-' 位置)
    CAR_CHAR_W = 46 #暫估, 用尺量出來的
    CAR_CHAR_H = 90 #暫估, 用尺量出來的
    CAR_SEP_W = 10 #暫估, 用尺量出來的
    CAR_BORDER = 5 # 暫估, 車牌邊寬
    
    CHAR_PATH = 'char_black'
    SEP_FILE = 'char_s/-.jpg'
    BORDER_FILE = 'car_plate_border.png'
    
    chars = None
    images = None
    border = None
    v = False #verbose
        
    def __init__(self, char_path, sep_file, border_file, v=False):
        c_p = self.CHAR_PATH if char_path is None else char_path
        s_f = self.SEP_FILE if sep_file is None else sep_file
        b_f = self.BORDER_FILE if border_file is None else border_file
        self.v = v
        
        # Load chars & seperator
        self._preload_chars(c_p, s_f)

        # load plate border
        b = cv2.imread(b_f)
        self.border = cv2.resize(b,(self.CAR_PLATE_W, self.CAR_PLATE_H))
    
    def _preload_chars(self, path, sep_fn):
        self.images = []
        self.chars = []
        files = glob(path + '/*.[jp][pn]g')
        for f in files:
            img = cv2.imread(f)
            img = cv2.resize(img, (self.CAR_CHAR_W, self.CAR_CHAR_H))
            self.images.append(img)
            self.chars.append(f[-5])
    
        img = cv2.imread(sep_fn)
        img = cv2.resize(img, (self.CAR_SEP_W, self.CAR_CHAR_H))
        self.images.append(img)
        self.chars.append(sep_fn[-5])
        
        if self.v:
            print(f'{len(self.chars)} chars {self.chars} is loadeded')
        
    def gen_plate(self, num=1, method='affine'):
        plates = []
        labels = []
        content_w = self.CAR_PLATE_W-self.CAR_BORDER*2 #字面寬，不算車牌邊
        content_start = int((self.CAR_PLATE_H-self.CAR_CHAR_H)/2) #字開始高度
        spacing = int((content_w-self.CAR_CHAR_W*self.CAR_CHAR_AMT-self.CAR_SEP_W) / (self.CAR_CHAR_AMT+2)) #字與字間隔
        first = int((content_w-self.CAR_CHAR_W*self.CAR_CHAR_AMT-self.CAR_SEP_W-spacing*self.CAR_CHAR_AMT) / 2) #頭尾字與旁邊間隔
        
        for n in range(num):
            content = np.full((self.CAR_CHAR_H, content_w, 3), 255, dtype=np.uint8)
            cs = choices(range(len(self.chars)-1), k=self.CAR_CHAR_AMT) # chars 最後一個是 seperator
            p = first
            s = ''
            for i, c in enumerate(cs):
                #seperator
                if i == self.CAR_SEP_POS:
                    s += self.chars[-1]
                    content[:,p:p+self.CAR_SEP_W,:] = self.images[-1]
                    p += (self.CAR_SEP_W+spacing)
        
                #chars
                s += self.chars[c]
                content[:,p:p+self.CAR_CHAR_W,:] = self.images[c]
                p += (self.CAR_CHAR_W+spacing)
        
            labels.append(s)
            
            # 車牌組裝與變化
            content[content==0] = randint(0, 150) #變化字的顏色
            lp = self._screw(self.border) #底牌加螺絲
            lp[content_start:content_start+self.CAR_CHAR_H,self.CAR_BORDER:-self.CAR_BORDER,:] = content #字組底牌
            lp_padding = self._pad(lp) # padding
            m = self._affine(lp_padding) if method == 'affine' else self._perspective(lp_padding) #變形
            m = self._noise(m) #雜訊
            m = self._blur(m) #模糊
            m = self._brighten(m) #亮度
            m = self._crop(m) #裁切, 偏移
            plates.append(m)
    
        if self.v:
            print(f'{len(plates):,} car plates is generated')
    
        return plates, labels
    
    # padding
    def _pad(self, img):
        r = choices(range(0,255+1), k=6)
        padding_color1 = (r[0], r[1], r[2])
        padding_color2 = (r[3], r[4], r[5])
        PAD_W = self.CAR_PLATE_W+4 # 兩邊各補兩個 bit 讓顏色可以在 BORDER_REPLICATE on affine or perspective 作用
        PAD_H = int(PAD_W/2)
        lp_padding = np.full((PAD_H, PAD_W, 3), padding_color1, dtype=np.uint8)
        if randint(0,1): # 決定是否模擬分隔線
            r = int(PAD_H/5) #定義分隔線可能出現的上下界，避免太高或太低，以免沒隔過車牌
            pos = randint(r, PAD_H-r)
            lp_padding[:pos,:,:] = padding_color2
        lp_padding[int(self.CAR_PLATE_P/2)+1:int(self.CAR_PLATE_P/2)+1+self.CAR_PLATE_H, 2:-2, :] = img
        
        return lp_padding
    
    #加螺絲
    def _screw(self, img): # 注意， call by reference, 不能直接改 img
        b = img.copy()
        k = randint(0,17)
        start_L = 85 #85~102
        start_R = 294 #294~277
        height = 20
        pos_L = (start_L+k, height)
        pos_R = (start_R-k, height)
        radio = 8
        color = randint(0,255) #(192,192,192) #silver
        cv2.circle(b, pos_L, radio, (color,color,color), -1) #cv2.circle 會直接改圖，不用 return
        cv2.circle(b, pos_R, radio, (color,color,color), -1) #(color,)*3 會比較慢
        
        return b
    
    #變形
    def _perspective(self, img, rh=0.3, rw=0.35):
        h, w, _ = img.shape
        
        pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
        dh = choices(range(0, int(h*rh)+1), k=4)
        dw = choices(range(0, int(w*rw)+1), k=4)
        pts2 = np.float32([[dw[0], dh[0]], [dw[1], h-dh[1]], [w-dw[2], dh[2]], [w-dw[3], h-dh[3]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        per = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        return per
    
    #變形
    def _affine(self, img, begin=0, end=70):
        w, h, _ = img.shape
        pts1 = np.float32([[0, 0], [0, w], [h, 0]])
        r = choices(range(begin, end+1), k=6)
        pts2 = np.float32([[r[0], r[1]], [r[2], w-r[3]], [h-r[4], r[5]]])
        M = cv2.getAffineTransform(pts1, pts2)
        aff = cv2.warpAffine(img, M, (h, w), borderMode=cv2.BORDER_REPLICATE)
    
        return aff
    
    #模糊
    def _blur(self, img):
        blur_value = randint(0, 5) * 2 + 1
        blu = cv2.blur(img, (blur_value, blur_value))
        
        return blu
    
    #雜訊
    def _noise(self, img):
        m = img + np.random.normal(0, 20, img.shape)
        m[m<0] = 0
        m[m>255] = 255
        
        return m.astype(np.uint8)
    
    # 亮度
    def _brighten(self, img):
        bri = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bri = np.array(bri, dtype=np.float64)
        random_bright = .4 + np.random.uniform()
        bri[:, :, 2] = bri[:, :, 2] * random_bright
        bri[:, :, 2][bri[:, :, 2] > 255] = 255
        bri = np.array(bri, dtype=np.uint8)
        bri = cv2.cvtColor(bri, cv2.COLOR_HSV2BGR)
        
        return bri
    
    # 裁切, 至多 20%; 偏移
    def _crop(self, img):
        f = int(img.shape[0]/10)
        [dh1, dh2] = choices(range(0, f+1), k=2)
        dw = (dh1+dh2)*2 #為維持 1:2
        dw1 = randint(0, dh1+dh2)
        dw2 = dw-dw1
        cro = img[dh1:img.shape[0]-dh2, dw1:img.shape[1]-dw2]
        
        return cro


# In[97]:


if __name__ == "__main__":
    import os
    import argparse
    from time import time
    from sys import argv
    
    now = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="output plate path", type=str, default="./temp")
    parser.add_argument("-n", "--num", help="output plate number", type=int, default=1)
    # 避掉 jupyter notebook exception
    if argv[0][-21:] == 'ipykernel_launcher.py':
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    path = args.path
    if not os.path.isdir(path):
        os.mkdir(path)
    g = TaiwanLicensePlateGenerator('char_black', 'char_s/-.jpg', 'border/car_plate_border.png', True)
    plates, labels = g.gen_plate(args.num, 'perspective')
    for plate, label in zip(plates, labels):
        cv2.imwrite(path+'/'+label+'.jpg', plate)
    print(f'output to {path} for overall {time()-now:.2f} secs')


# In[ ]:




