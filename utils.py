from dataloader import *
def iou(pred, target):
    pred = pred.cuda()
    target = target.cuda()
    ious_tp = []
    ious_fp = []
    ious_fn = []
    #pred = pred.numpy(), 
    for cls in range(34):
        # Complete this function
        

        TP = np.sum(np.argwhere(pred==target==cls,1,0))
        FP = np.sum(np.argwhere(pred==cls!=target,1,0))
        FN = np.sum(np.argwhere(pred!=cls and target==cls,1,0))
        intersection = TP # intersection calculation
        union = TP + FP +FN # Union calculation
        if union == 0:
            ious_tp.append(float('nan'))
            ious_fp.append(float('nan'))
            ious_fn.append(float('nan'))
            # if there is no ground truth, do not include in evaluation
        else:
            ious_tp.append(TP)
            ious_fp.append(FP)
            ious_fn.append(FN)
                #ious.append(intersection/union)# Append the calculated IoU to the list ious
    return ious_tp, ious_fp, ious_fn


def pixel_acc(pred, target):
    pass
    #Complete this function
    # included in main