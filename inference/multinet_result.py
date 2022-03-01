import numpy as np
import tensorflow as tf

def get_voice_vote_nms(all_preds_class, all_preds_bbox):
    good_pred_thresh = 0.5
    bad_pred_thresh = 0.3
    uncertain_indices = np.where(np.logical_and(all_preds_class > bad_pred_thresh, all_preds_class < good_pred_thresh))
    uncertain_indices = np.where(np.logical_and(all_preds_class > bad_pred_thresh, all_preds_class < good_pred_thresh))
    model_preds = np.where(all_preds_class >= good_pred_thresh, 1, 0)
    model_preds[uncertain_indices[0], uncertain_indices[1]] = 2

    majority_vote = 4
    voice_vote = []
    num_good_bad = []
    for ap in all_preds_class:    
        bad_pred = np.sum(ap < bad_pred_thresh)
        good_pred = np.sum(ap >= good_pred_thresh)
        if bad_pred >= majority_vote :
            voice_vote.append(0)
        elif good_pred >= majority_vote:
            voice_vote.append(1)
        else:
            voice_vote.append(2)

    bbox_nms_pred = []
    for class_scores, bbox in zip(all_preds_class, all_preds_bbox):
        bbox = np.hstack([bbox[:,i].reshape(-1,1) for i in [1,0,3,2]])
        selected_indices = tf.image.non_max_suppression(bbox, class_scores, 1, score_threshold=good_pred_thresh)
        selected_boxes = tf.gather(bbox, selected_indices)
        tmp = selected_boxes.numpy().squeeze()
        bbox_nms_pred.append(tmp if(tmp.size == 4) else [0]*4) #if no boxes are selected, make nms pred to be zero
    
    return voice_vote, bbox_nms_pred

def get_bbox_upscaled(bbox_coord, w, h):
    (startX, startY, endX, endY) = bbox_coord
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    return startX, startY, endX, endY