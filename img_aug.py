from random import  randint
import numpy as np
import  cv2
def im_warp_perspective(imgs,keypoints,limb_points,mb = 10):
    # 4.perspective
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]
    ra = randint
    pts3 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts4 = np.float32([[ra(0,mb),ra(0,mb)],[w-ra(0,mb),ra(0,mb)],[ra(0,mb),h-ra(0,mb)],[w-ra(0,mb),h-ra(0,mb)]])
    M_perspective = cv2.getPerspectiveTransform(pts3,pts4)
    imgs_r = []
    for img in imgs:
        img_warp = cv2.warpPerspective(img, M_perspective, (w, h))
        imgs_r.append(img_warp)
    keypoints_r = []
    for pard_id,x,y in keypoints:
        p0 = np.array([x,y]).astype(np.float32).reshape(1,1,2)
        p = cv2.perspectiveTransform(p0,M_perspective).reshape(1,-1)
        x = p[0,0]
        y = p[0,1]
        keypoints_r.append([pard_id,x,y])
    limb_r = []
    for limb_id,x0,y0,x1,y1 in limb_points:
        p0 = np.array([x0,y0]).astype(np.float32).reshape(1,1,2)
        p1 = np.array([x1,y1]).astype(np.float32).reshape(1,1,2)
        p = np.concatenate([p0,p1],axis=0)
        p = cv2.perspectiveTransform(p,M_perspective).reshape(2,-1)
        x0 = p[0,0]
        y0 = p[0,1]
        x1 = p[1,0]
        y1 = p[1,1]

        limb_r.append([limb_id,x0,y0,x1,y1])
    return imgs_r,keypoints_r,limb_r
def im_aug(imgs,keypoints,parts):
    imgs,keypoints,parts =  im_rotate(imgs,keypoints,parts)
    # return imgs,keypoints,parts
    return  im_warp_perspective(imgs,keypoints,parts,int(imgs[0].shape[1]//6))
def im_rotate(imgs,keypoints,limb_points):
    angle = randint(-15,15)
    matrix = cv2.getRotationMatrix2D((imgs[0].shape[1]//2.,imgs[0].shape[0]/2.),angle,1.0);
    imgs_r = []
    for img in imgs:
        imgs_r.append( cv2.warpAffine(img,matrix,(imgs[0].shape[1],imgs[0].shape[0])))
    keypoints_r = []
    limb_r = []
    keypoints_r = []
    for pard_id,x,y in keypoints:
        p0 = np.array([x,y]).astype(np.float32).reshape(1,1,2)
        p = cv2.transform(p0,matrix).reshape(1,-1)
        # print(p.shape)
        x = p[0,0]
        y = p[0,1]
        keypoints_r.append([pard_id,x,y])
    limb_r = []
    for limb_id,x0,y0,x1,y1 in limb_points:
        p0 = np.array([x0,y0]).astype(np.float32).reshape(1,1,2)
        p1 = np.array([x1,y1]).astype(np.float32).reshape(1,1,2)
        p = np.concatenate([p0,p1],axis=0)
        p = cv2.transform(p,matrix).reshape(2,-1)
        x0 = p[0,0]
        y0 = p[0,1]
        x1 = p[1,0]
        y1 = p[1,1]

        limb_r.append([limb_id,x0,y0,x1,y1])
    return imgs_r,keypoints_r,limb_r



