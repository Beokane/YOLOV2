import numpy as np


def compute_iou(anchor_boxes, gt_box):
    """计算先验框和真实框之间的IoU
    Input: \n
        anchor_boxes: [K, 4] \n
            gt_box: [1, 4] \n
    Output: \n
                iou : [K,] \n
    """

    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    # 计算先验框的左上角点坐标和右下角点坐标
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # xmin
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # ymin
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # xmax
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # ymax
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # 我们将真实框扩展成[K, 4], 便于计算IoU. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)
    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    # 计算真实框的左上角点坐标和右下角点坐标
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # xmin
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # ymin
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # xmax
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # ymin
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # 计算IoU
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """将输入进来的只包含wh的先验框尺寸转换成[N, 4]的ndarray类型,
       包含先验框的中心点坐标和宽高wh,中心点坐标设为0. \n
    Input: \n
        anchor_size: list -> [[h_1, w_1],  \n
                              [h_2, w_2],  \n
                               ...,  \n
                              [h_n, w_n]]. \n
    Output: \n
        anchor_boxes: ndarray -> [[0, 0, anchor_w, anchor_h], \n
                                  [0, 0, anchor_w, anchor_h], \n
                                  ... \n
                                  [0, 0, anchor_w, anchor_h]]. \n
    """
    anchor_number = len(anchor_size)
    anchor_boxes = np.zeros([anchor_number, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    return anchor_boxes

def generate_txtytwth(gt_label, w, h, s, anchor_size, ignore_thresh):
    '''
    在真实框所在的窗格内，将先验框标记为正样本和忽略样本。
    负样本默认在没有真实框出现的网格上(因为初始化的时候都是0)
    '''
    # gt_label: [xmin, ymin, xmax, ymax, class_id]
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算真实边界框的中心点和宽高
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h
    if box_w < 1e-4 or box_h < 1e-4:
        # print('not a valid data !!!')
        return False    
    # 将真是边界框的尺寸映射到网格的尺度上去
    c_x_s = c_x / s
    c_y_s = c_y / s
    box_ws = box_w / s
    box_hs = box_h / s
    # 计算中心点所落在的网格的坐标
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 忽略中心点的偏移，直接使用标签和先验框的宽高匹配策略。后续可通过回归方式修正。
    anchor_boxes = set_anchors(anchor_size)
    gt_box = np.array([[0, 0, box_ws, box_hs]])
    iou = compute_iou(anchor_boxes, gt_box)
    # 只保留大于ignore_thresh的先验框去做正样本匹配,
    iou_mask = (iou > ignore_thresh)
    result = []
    if iou_mask.sum() == 0:
        # 如果所有的先验框算出的IoU都小于阈值，那么就将IoU最大的那个先验框分配给正样本.
        # 其他的先验框统统视为负样本
        index = np.argmax(iou)
        p_w, p_h = anchor_size[index]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_ws / p_w)
        th = np.log(box_hs / p_h)
        weight = 2.0 - (box_w / w) * (box_h / h)
        # index 是先验框的索引（0-4）
        # grid_x, grid_y 是中心点所在的网格坐标（13，13）
        # tx, ty 是中心点的偏移量
        # tw, th 是标签的相对网格宽高与最匹配先验框的网格宽高的比值的对数
        # weight 是权重， 2.0 - 归一化标签的面积
        # xmin, ymin, xmax, ymax 是归一化后角点坐标
        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])    
        return result  
    else:
        # 有至少一个先验框的IoU超过了阈值.
        # 但我们只保留超过阈值的那些先验框中IoU最大的，其他的先验框忽略掉，不参与loss计算。
        # 而小于阈值的先验框统统视为负样本。
        best_index = np.argmax(iou)
        for index, iou_m in enumerate(iou_mask):
            if iou_m:
                if index == best_index:
                    p_w, p_h = anchor_size[index]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = np.log(box_ws / p_w)
                    th = np.log(box_hs / p_h)
                    weight = 2.0 - (box_w / w) * (box_h / h)
                    result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                else:
                    # 对于被忽略的先验框，我们将其权重weight设置为-1
                    result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1.0, 0., 0., 0., 0.])
        return result 

def gt_creator(input_size, stride, label_lists, anchor_size, ignore_thresh):
    '''
    相当于给每张图像上的每个网格的每个先验框附上标签，包括标签的类别、是否为正样本、
    中心点的偏移量、宽高的偏移量、权重等。
    '''
    # 必要的参数
    # label_Lists shape: batch_size, num_bbox, 4+1
    batch_size = len(label_lists)
    s = stride
    w = input_size
    h = input_size
    ws = w // s
    hs = h // s
    # anchor_size shape: achor_size, 2
    anchor_number = len(anchor_size)
    gt_tensor = np.zeros([batch_size, hs, ws, anchor_number, 1+1+4+1+4])

    # 制作正样本
    for batch_index in range(batch_size):
        # 对于没有遍历到的网格，默认其obj标签为0，即背景，为负样本。
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            gt_class = int(gt_label[-1])
            results = generate_txtytwth(gt_label, w, h, s, anchor_size, ignore_thresh)
            if results:
                for result in results:
                    index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = result
                    if weight > 0.:
                        if grid_y < gt_tensor.shape[1] and grid_x < gt_tensor.shape[2]:
                            gt_tensor[batch_index, grid_y, grid_x, index, 0] = 1.0
                            gt_tensor[batch_index, grid_y, grid_x, index, 1] = gt_class
                            gt_tensor[batch_index, grid_y, grid_x, index, 2:6] = np.array([tx, ty, tw, th])
                            gt_tensor[batch_index, grid_y, grid_x, index, 6] = weight
                            gt_tensor[batch_index, grid_y, grid_x, index, 7:] = np.array([xmin, ymin, xmax, ymax])
                    else:
                        # 对于那些被忽略的先验框，其gt_obj参数为-1，weight权重也是-1
                        gt_tensor[batch_index, grid_y, grid_x, index, 0] = -1.0
                        gt_tensor[batch_index, grid_y, grid_x, index, 6] = -1.0
    gt_tensor = gt_tensor.reshape(batch_size, hs * ws * anchor_number, 1+1+4+1+4)
    return gt_tensor