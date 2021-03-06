'''
Created on Jan 16, 2018

@author: kohill
'''
from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2,os,random
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
class DataIter(Dataset):
    def __init__(self):
        annFile = '/home/kohill/hszc/data/coco/annotations/person_keypoints_train2014.json' # keypoint file
        self.trainimagepath = '/home/kohill/hszc/data/coco/train2014'             # train image path        
        coco = COCO(annFile)
        catIds = coco.getCatIds(catNms=['person']);
        self.catIds = coco.getCatIds(catNms=['person'])
        self.imgIds = coco.getImgIds(catIds=catIds );  
        self.coco_kps = coco
        self.NUM_PARTS=18
        self.NUM_LINKS=19
        self.HEAT_RADIUS = 12
        self.PART_LINE_WIDTH=16
    def __len__(self):
        return len(self.imgIds)
    def __getitem__(self,index):
        img = self.coco_kps.loadImgs(self.imgIds[index])[0]
        img_ori = cv2.imread(os.path.join(self.trainimagepath , img['file_name']))
        img_human_seg = np.zeros(shape = img_ori.shape[0:2],dtype=np.float32)
        loss_mask = np.ones_like(img_human_seg)
        annIds = self.coco_kps.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
        anns = self.coco_kps.loadAnns(annIds)
#         plt.imshow(img_ori)
#         self.coco_kps.showAnns(anns)
#         plt.show()
        assert len(anns) > 0
        assert 'segmentation' in anns[0] or 'keypoints' in anns[0]
        polygons = []
        color = []
        keypoints = [] #(part_id,x,y)
        parts = []#((partid0,x0,y0),(partid1,x1,y1))
        for ann in anns:
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        cv2.drawContours(img_human_seg,[poly[np.newaxis,:].astype(np.int32)],0,(1,1,1),-1)
                        polygons.append(Polygon(poly))
                        color.append(c)
                    if 'keypoints' in ann and (ann['num_keypoints'] < 5 or ann['area']< 32*32) :
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            cv2.drawContours(loss_mask,[poly[np.newaxis,:].astype(np.int32)],0,(0,0,0),-1)

                else:
                    # mask
                    t = self.coco_kps.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                    m = maskUtils.decode(rle)

                    loss_mask *= (1.0-m[:,:,0]).astype(np.float32)

            COCO_to_ours_1 = [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
            COCO_to_ours_2 = [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
            mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
            mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]   
            assert len(COCO_to_ours_1) == len(COCO_to_ours_2) == self.NUM_PARTS                 
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
#                 sks = np.array(self.coco_kps.loadCats(ann['category_id'])[0]['skeleton'])-1
                kp = np.array(ann['keypoints'])

                x_coco = kp[0::3]
                y_coco = kp[1::3]
                v_coco = kp[2::3]
                x = []
                y = []
                v = []
                for index1,index2 in zip(COCO_to_ours_1,COCO_to_ours_2):
                    index1 -= 1
                    index2 -= 1
                    x.append(0.5*(x_coco[index1] + x_coco[index2]))
                    y.append(0.5*(y_coco[index1] + y_coco[index2]))
                    v.append(min(v_coco[index1],v_coco[index2]))
                for i in range(self.NUM_PARTS):
                    if v[i] > 0:
                        # cv2.circle(heatmaps[i],(int(round(x[i])),int(round(y[i]))),self.HEAT_RADIUS,(1,1,1),-1)
                        keypoints.append([i,x[i],y[i]])
                for i in range(self.NUM_LINKS):
                    kp0,kp1 = mid_1[i]-1,mid_2[i]-1
                    if v[kp0] > 0 and v[kp1] > 0:
                        parts.append([i,x[kp0],y[kp0],x[kp1],y[kp1]])
        if len(img_ori.shape) == 2:
            temp = np.empty(shape= (img_ori.shape[0],img_ori.shape[1],3),dtype = np.uint8)
            for i in range(3):
                temp[:,:,i] = img_ori 
            print('gray img')           


        from img_aug import im_aug
        [img_ori,loss_mask],keypoints,parts = im_aug([img_ori,loss_mask],keypoints,parts)
        # for pard_id,x,y in keypoints:
        #     cv2.circle(heatmaps[pard_id],
        #                (int(round(x)),int(round(y))),self.HEAT_RADIUS,(1,1,1),-1)
        #

        img_ori = np.transpose(img_ori,(2,0,1))
        # heatmaps = np.array(heatmaps)

        # heatmaps = np.concatenate([heatmaps,np.max(heatmaps,axis = 0)[np.newaxis,:,:],img_human_seg[np.newaxis,:,:]])
        # heatmaps = np.concatenate([heatmaps,img_human_seg[np.newaxis]])
        # heatmaps = np.concatenate([heatmaps,np.min(heatmaps,axis=0)[np.newaxis]])
        loss_mask = loss_mask[np.newaxis,:,:]
        
        img_ori,loss_mask = self.im_transpose([img_ori, loss_mask], axes=(1,2,0))
        img_ori,keypoints,parts,loss_mask = self.im_resize(img_ori, keypoints, parts, loss_mask)
        # img_ori,heatmaps,pafmaps,loss_mask = self.im_crop(img_ori, heatmaps, pafmaps, loss_mask)
        stride = 8.0
        heatmaps = [np.zeros(shape = (46,46),dtype=np.float32) for _ in range(self.NUM_PARTS)]
        # pafmaps = [np.zeros(shape = (46,46),dtype=np.float32) for _ in range(self.NUM_LINKS*2)]

        for m in range(int(368//stride)):
            for n in range(int(368//stride)):
                ori_x = n *stride + stride / 2 - 0.5
                ori_y = m * stride + stride / 2 - 0.5
                for  pard_id,x,y in keypoints:
                    d2 = (ori_x-x)**2+(ori_y-y)**2
                    thgma  = 7.0
                    exponent = d2 / 2.0 / (thgma**2)
                    heatmaps[pard_id][m, n] = max(np.exp(-exponent), heatmaps[pard_id][m,n])
        # for limb_id,x0,y0,x1,y1 in parts:
        #     vec = np.array([x0,y0])-np.array([x1,y1])
        #     vec /= np.linalg.norm(vec) + 0.0001
        #     mask_ = np.zeros_like(loss_mask,dtype = np.uint8)
        #     cv2.line(mask_,(int(round(x0)),int(round(y0))),
        #              (int(round(x1)),int(round(y1))),(1,1,1),self.PART_LINE_WIDTH)
        #     pafmaps[limb_id *2] = np.squeeze( mask_[::int(stride),::int(stride)]) * vec[0]
        #     pafmaps[limb_id *2 + 1] = np.squeeze( mask_[::int(stride),::int(stride)]) * vec[1]
        pafmaps  = [np.zeros_like(np.squeeze(loss_mask)) for _ in range(self.NUM_LINKS * 2)]
        pafmaps_count = [np.zeros_like(np.squeeze(loss_mask)) for _ in range(self.NUM_LINKS * 2)]

        for limb_id,x0,y0,x1,y1 in parts:
            p0 = np.array([x0,y0])
            p1 = np.array([x1,y1])
            mask_ = np.zeros_like(np.squeeze(loss_mask),dtype = np.uint8)
            cv2.line(mask_,(int(round(x0)),int(round(y0))),
                     (int(round(x1)),int(round(y1))),(1,1,1),self.PART_LINE_WIDTH)
            vec = p1 -p0
            vec =  vec/(np.linalg.norm(vec)+0.001)
            vec_index = np.where(np.squeeze(mask_) )
            pafmaps[2*limb_id][vec_index] += vec[0]
            pafmaps[2*limb_id + 1][vec_index] += vec[1]
            pafmaps_count[2*limb_id][vec_index] += 1
            pafmaps_count[2*limb_id+1][vec_index] += 1

        pafmaps_count = np.array(pafmaps_count)
        pafmaps = np.array(pafmaps)
        pafmaps[np.where(pafmaps_count!=0)] /= pafmaps_count[np.where(pafmaps_count!=0)]
        pafmaps = pafmaps[:,::int(stride),::int(stride)]
        heatmaps = np.array(heatmaps)
        heatmaps = np.concatenate([heatmaps,np.min(heatmaps,axis=0)[np.newaxis]])
        dest_size = (46,46)
        # heatmaps = cv2.resize(heatmaps,dest_size,interpolation=cv2.INTER_CUBIC)
#         heatmaps = cv2.GaussianBlur(heatmaps,(3,3),1)
#         pafmaps = cv2.resize(pafmaps,dest_size)
        loss_mask = cv2.resize(loss_mask,dest_size)
        loss_mask = loss_mask[:,:,np.newaxis]
        img_ori,loss_mask = self.im_transpose([img_ori, loss_mask], axes=(2,0,1))

        loss_mask = np.min(loss_mask,axis = 0)[np.newaxis,:,:]

        # heatmaps[np.where(heatmaps>=0.1)] = 0.999
        heatmaps[np.where(heatmaps<0)] = 0
        loss_mask[np.where(loss_mask < 0.5)] = 0
        # print("res",loss_mask.shape)
        # print("pafmap.shape",pafmaps.shape)
        return img_ori,heatmaps,pafmaps,loss_mask

    def im_transpose(self,imgs,axes = (2,1,0)):
        imgs_r = []
        for img in  imgs:
            imgs_r.append(np.transpose(img,axes=axes))
        return imgs_r
        
    def im_resize(self,img_ori,keypoints,parts,loss_mask):
        fscale = 368.0/np.max(img_ori.shape[0:2])
        
        img_ori,loss_mask = list(
            map(lambda x:cv2.resize(x,(
                int(np.round(fscale*img_ori.shape[1])),int(np.round(fscale*img_ori.shape[0])))
                ),
                [img_ori,loss_mask]))
        loss_mask = loss_mask[:,:,np.newaxis]
        def img_pad(x):
            sm = np.max(x.shape[0:2])
            padded = np.zeros(shape = (sm,sm,x.shape[2]),dtype=np.float32)
            padded[:x.shape[0],:x.shape[1]] = x
            return padded
        img_ori_padded = img_pad(img_ori)
        # heatmaps_padded = img_pad(heatmaps)
        loss_mask_padded = img_pad(loss_mask)
        keypoints_r = []
        for part_id,x,y in keypoints:
            keypoints_r.append([part_id,x*fscale,y*fscale])
        parts_r = []
        for limb_id,x0,y0,x1,y1 in parts:
            parts_r.append([limb_id,x0*fscale,y0*fscale,x1* fscale,y1*fscale])
        return img_ori_padded,keypoints_r,parts_r,loss_mask_padded
    def im_crop(self,img_ori,heatmaps,pafmaps,loss_mask):
        start_m = random.randint(0,img_ori.shape[0]-368)
        start_n = random.randint(0,img_ori.shape[1]-368)
        img_ori,heatmaps,pafmaps,loss_mask = list(
            map(lambda x:x[start_m:(start_m+368),start_n:(start_n+368)],
                [img_ori,heatmaps,pafmaps,loss_mask]))
        return img_ori,heatmaps,pafmaps,loss_mask
def draw_heatmap(heatmap,img=None):
    _,axes = plt.subplots(4,5,figsize=(35,28))
    plt.subplots_adjust(wspace = 0,hspace = 0.15)
    for i in range(5):
        for j in range(4):
            index = i*5+j                
            if index < heatmap.shape[0]:
#                 heatmap[index,:,:][0,0] = 1
                axes[j][i].imshow(heatmap[index,:,:])
            elif index == heatmap.shape[0]:
                axes[j][i].title.set_text("max")
                axes[j][i].imshow(np.max(heatmap,axis = 0))                    
            elif img is not None and index == heatmap.shape[0]+1:
                axes[j][i].title.set_text("img")
                axes[j][i].imshow(img)
            else:
                axes[j][i].imshow(heatmap[-1,:,:])
def collate_fn(batch):
    imgs_batch = []
    heatmaps_batch = []
    pafmaps_batch = []
    loss_mask_batch = []
    for sample in batch:
        img_ori,heatmaps,pafmaps,loss_mask = sample
        imgs_batch.append(img_ori[np.newaxis])
        heatmaps_batch.append(heatmaps[np.newaxis])
        pafmaps_batch.append(pafmaps[np.newaxis])
        loss_mask_batch.append(loss_mask[np.newaxis])
    imgs_batch =np.concatenate(imgs_batch,axis = 0)
    heatmaps_batch =np.concatenate(heatmaps_batch,axis = 0)
    pafmaps_batch =np.concatenate(pafmaps_batch,axis = 0)
    loss_mask_batch =np.concatenate(loss_mask_batch,axis = 0)            
    return [imgs_batch,heatmaps_batch,pafmaps_batch,loss_mask_batch]    
def getDataLoader(batch_size = 16):
    test_iter = DataIter()
    r = DataLoader(test_iter, batch_size=batch_size, shuffle=True, num_workers=10, collate_fn=collate_fn, pin_memory=False,drop_last = True)
    return r
if __name__ == '__main__':
    print("length",len(getDataLoader(8)))
    data_iter = DataIter()
    for i in range(len(data_iter)):
        da = data_iter[i]
        for d in da:
            print(d.shape)

        x = list(map(lambda x: np.transpose(x,(1,2,0)) if len(x.shape) > 2 else x, da))
        
        fig, axes = plt.subplots(2, len(x)//2 + len(x)%2, figsize=(45, 45),
                             subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(hspace=0.3, wspace=0.05) 
 
        count = 0
       
        for j in range(len(axes)):
            for i in range(len(axes[0])):
                try:
                    img = x[count]
                    count += 1
                except IndexError:
                    break
                print(count,len(x))
                if len(img.shape)>2 and img.shape[2]==38:
                    img = np.array([np.sqrt(img[:,:,k *2] ** 2 + img[:,:,k *2+1 ] ** 2) for k in range(img.shape[2]//2)])
                    axes[j][i].imshow(np.max(img,axis = 0))
                    print("limb")
                elif len(img.shape)>2 and img.shape[2] > 3:
                    axes[j][i].imshow(np.max(img[:,:,:-1],axis = 2)) 
                elif len(img.shape)>2 and img.shape[2] == 1:
                    axes[j][i].imshow(img[:,:,0]) 
                else:
                    axes[j][i].imshow(img.astype(np.uint8))
                     
        plt.show()
    pass