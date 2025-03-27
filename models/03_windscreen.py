import cv2
import numpy as np

def Brown_Conrady_Cardi_distortion(x_idx, y_idx, k1, k2, k3, k4, k5, k6, p1, p2):
    r_idx = np.hypot(x_idx,y_idx)
    sq_x_idx = np.square(x_idx)
    sq_y_idx = np.square(y_idx)
    sq_r_idx = sq_x_idx + sq_y_idx
    sqsq_r_idx = np.square(sq_r_idx)
    sqtr_r_idx = np.square(sq_r_idx*r_idx)
    BC_x_idx = x_idx*np.divide((1+k1*sq_r_idx+k2*sqsq_r_idx+k3*sqtr_r_idx),
                               (1+k4*sq_r_idx+k5*sqsq_r_idx+k6*sqtr_r_idx))+2*p1*x_idx*y_idx+p2*(sq_r_idx+2*sq_x_idx)
    BC_y_idx = y_idx*np.divide((1+k1*sq_r_idx+k2*sqsq_r_idx+k3*sqtr_r_idx),
                               (1+k4*sq_r_idx+k5*sqsq_r_idx+k6*sqtr_r_idx))+p1*(sq_r_idx+2*sq_y_idx)+2*p2*x_idx*y_idx
    
    return BC_x_idx, BC_y_idx


def distortion(input_img):
    # Define fixed variables
    (output_img_height, output_img_width) = input_img.shape[:2]
    (input_height, input_width) = input_img.shape[:2]
    PADDING_TOP = 0
    PADDING_BOTTOM = 100
    PADDING_LEFT = 0
    PADDING_RIGHT = 0 

    # Brown Conrandy Cardi parameters
    # K1 = 0.0000006
    K1 = 0.0000001
    K2 = 0
    K3 = 0
    K4 = 0
    K5 = 0
    K6 = 0
    P1 = 0
    P2 = 0

    if len(input_img.shape)==2:
        canvas = np.zeros((input_height+PADDING_TOP+PADDING_BOTTOM,input_width+PADDING_LEFT+PADDING_RIGHT), dtype=np.uint8)
        canvas[PADDING_TOP:PADDING_TOP+input_height,PADDING_LEFT:PADDING_LEFT+input_width] = input_img
        canvas_stage = np.zeros((canvas.shape[0]*3,canvas.shape[1]*3), dtype=np.uint8)
        img_h_c = canvas.shape[0] #720 = half of canvas height: 1440 by default
        img_w_c = canvas.shape[1] #960 = half of canvas width: 1920 by default
        canvas_stage[img_h_c:2*img_h_c,img_w_c:2*img_w_c] = canvas
    else:
        canvas = np.zeros((input_height+PADDING_TOP+PADDING_BOTTOM,input_width+PADDING_LEFT+PADDING_RIGHT,input_img.shape[2]), dtype=np.uint8)
        canvas[PADDING_TOP:PADDING_TOP+input_height,PADDING_LEFT:PADDING_LEFT+input_width,:] = input_img
        canvas_stage = np.zeros((canvas.shape[0]*3,canvas.shape[1]*3,input_img.shape[2]), dtype=np.uint8)
        img_h_c = canvas.shape[0] #720 = half of canvas height: 1440
        img_w_c = canvas.shape[1] #960 = half of canvas width: 1920
        canvas_stage[img_h_c:2*img_h_c,img_w_c:2*img_w_c,:] = canvas

    col_idxInRow = np.arange(0,0.5*output_img_width,1,dtype=np.float64)
    col_idxQuaterCanv = np.tile(col_idxInRow, (int(0.5*output_img_height),1))
    row_idxInCol = np.arange(0,0.5*output_img_height,1,dtype=np.float64)
    row_idxQuaterCanv = np.transpose(np.tile(row_idxInCol, (int(0.5*output_img_width),1)))
    
    BC_col_idxQuaterCanv, BC_row_idxQuaterCanv =  Brown_Conrady_Cardi_distortion( col_idxQuaterCanv, row_idxQuaterCanv, 
                                                                              K1, #default 0.0000006, can be changed if parameters are available by calibration.
                                                                              # if image is a quater the full size, 
                                                                              # k1 should be in 4x to achieve the same distortion
                                                                              K2, #default 0, can be changed if parameters are available by calibration.
                                                                              K3, #default 0, can be changed if parameters are available by calibration.
                                                                              K4, #default 0, can be changed if parameters are available by calibration.
                                                                              K5, #default 0, can be changed if parameters are available by calibration.
                                                                              K6, #default 0, can be changed if parameters are available by calibration.
                                                                              P1, #default 0, can be changed if parameters are available by calibration.
                                                                              P2) #default 0, can be changed if parameters are available by calibration.
    
    revH_BC_col_idxQuaterCanv = np.flip(BC_col_idxQuaterCanv, axis=1)
    revH_BC_row_idxQuaterCanv = np.flip(BC_row_idxQuaterCanv, axis=1)

    revV_BC_col_idxQuaterCanv = np.flip(BC_col_idxQuaterCanv, axis=0)
    revV_BC_row_idxQuaterCanv = np.flip(BC_row_idxQuaterCanv, axis=0)

    revD_BC_col_idxQuaterCanv = np.flip(np.flip(BC_col_idxQuaterCanv, axis=1), axis=0)
    revD_BC_row_idxQuaterCanv = np.flip(np.flip(BC_row_idxQuaterCanv, axis=1), axis=0)

    col_idxQuadIV = int(0.5*img_w_c) + BC_col_idxQuaterCanv
    row_idxQuadIV = int(0.5*img_h_c) + BC_row_idxQuaterCanv

    col_idxQuadIII = int(0.5*img_w_c) - 1 - revH_BC_col_idxQuaterCanv
    row_idxQuadIII = int(0.5*img_h_c) + revH_BC_row_idxQuaterCanv

    col_idxQuadII = int(0.5*img_w_c) - 1 - revD_BC_col_idxQuaterCanv
    row_idxQuadII = int(0.5*img_h_c) - 1 - revD_BC_row_idxQuaterCanv

    col_idxQuadI = int(0.5*img_w_c) + revV_BC_col_idxQuaterCanv
    row_idxQuadI = int(0.5*img_h_c) - 1 - revV_BC_row_idxQuaterCanv

    col_idxCanvas = np.concatenate((np.concatenate((col_idxQuadII,col_idxQuadI),axis=1),
                                    np.concatenate((col_idxQuadIII,col_idxQuadIV),axis=1))
                                    ,axis=0)

    row_idxCanvas = np.concatenate((np.concatenate((row_idxQuadII,row_idxQuadI),axis=1),
                                    np.concatenate((row_idxQuadIII,row_idxQuadIV),axis=1))
                                    ,axis=0)
    
    D1_col_idxCanvas = np.int64(np.reshape(col_idxCanvas,[1,-1]))
    D1_row_idxCanvas = np.int64(np.reshape(row_idxCanvas,[1,-1]))
    
    if len(input_img.shape)==2:
        samples_targetCanvas = np.reshape(canvas_stage[img_h_c+D1_row_idxCanvas,img_w_c+D1_col_idxCanvas],[output_img_height,output_img_width])
        samples_targetCanvas = samples_targetCanvas[35:35+1311, 105:105+1726]
    else:
        samples_targetCanvas = np.reshape(canvas_stage[img_h_c+D1_row_idxCanvas,img_w_c+D1_col_idxCanvas,:],[output_img_height,output_img_width,input_img.shape[2]])
        samples_targetCanvas = samples_targetCanvas[35:35+1311, 105:105+1726, :]
    
    return samples_targetCanvas