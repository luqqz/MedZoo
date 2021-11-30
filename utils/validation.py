import matplotlib as mpl
from matplotlib import cm
import nibabel
import os
import numpy
import matplotlib.pyplot as plt

task_mask = [
    6, # liver
    [2, 3], # kidney
    1 # spleen
]

root1="E:\\Studia\\MGR\\FINAL TRAINING\\DoDNet\\inference\\my\\0TEST"
root2="E:\\Studia\\MGR\\FINAL TRAINING\\DoDNet\\inference\\my\\1TEST"
root3="E:\\Studia\\MGR\\FINAL TRAINING\\DoDNet\\inference\\my\\2TEST"
file = "label0030_pred.nii.gz"
# pred_path_1="E:\\Studia\\MGR\\FINAL TRAINING\\DoDNet\\inference\\pretrained\\TASK1"
# pred_path_11="E:\\Studia\\MGR\\FINAL TRAINING\\DoDNet\\inference\\pretrained\\TASK11"
# pred_path_2="E:\\Studia\\MGR\\FINAL TRAINING\\DoDNet\\inference\\pretrained\\TASK2"
gt_path="E:\\Studia\\MGR\\code\\DoDNet\\dataset\\0123456_spacing_same\\1BTCV\\labelsTr\\kidney_normal"
# gt_3_076_076_path="E:\\Studia\\MGR\\code\\DoDNet\\dataset\\0123456_spacing_same\\1BTCV\\labelsTr\\kidney3"

global tpg
global fpg
global fng

global tp
global fp
global fn

tpg = 0
fpg = 0
fng = 0
dice = []

#
# Tasks
#  0 - Liver
#  1 - Kidney
#  2 (6) - Spleen
#  9 - All
#
TASK = 9

def dice_fn(predict, target):

    global tp
    global fp
    global fn
    ll = numpy.sum(predict[numpy.where(target == 1)])
    print("TP LOCAL: " + str(ll))
    tp += ll
    fpl = numpy.sum(predict[numpy.where(numpy.logical_and(target == 0, predict == 1))])
    fnl = numpy.sum(target[numpy.where(numpy.logical_and(target == 1, predict == 0))])
    fp += fpl
    fn += fnl
    print("FP LOCAL: " + str(fpl))
    print("FL LOCAL: " + str(fnl))
    ll *= 2
    mm = numpy.sum(predict) + numpy.sum(target)

    return mm


print(os.path.join(root1, file))
print(os.path.join(gt_path, file.replace("_pred", "")))
pred_seg_0 = nibabel.load(os.path.join(root, file)).get_fdata()
pred_seg_1 = nibabel.load(os.path.join(pred_path_1, file)).get_fdata()
pred_seg_2 = nibabel.load(os.path.join(pred_path_2, file)).get_fdata()
pred_seg = pred_seg_0 + pred_seg_1 + pred_seg_2
pred_seg1 = nibabel.load(os.path.join(root1, file)).get_fdata()
pred_seg2 = nibabel.load(os.path.join(root2, file)).get_fdata()
pred_seg3 = nibabel.load(os.path.join(root3, file)).get_fdata()
pred_seg = pred_seg1 + pred_seg2 + pred_seg3
pred_seg[pred_seg > 1] = 1
#pred_seg[pred_seg != 1] = 0
#pred_seg[pred_seg == 1] = 1
#pred_seg[pred_seg > 0] = 1
gt_seg = nibabel.load(os.path.join(gt_path, file.replace("_pred", ""))).get_fdata()
# import matplotlib.pyplot as plt
# plt.figure(0)
# plt.subplot(1,2,1)
# plt.imshow(pred_seg[:,:,100])
# plt.subplot(1,2,2)
# plt.imshow(gt_seg[:,:,100])
# plt.show()
if TASK == 9:
    gt_seg[gt_seg == task_mask[0]] = 1
    gt_seg[gt_seg == task_mask[1][1]] = 1
    gt_seg[gt_seg == task_mask[1][0]] = 1
    gt_seg[gt_seg == task_mask[2]] = 1
    gt_seg[gt_seg != 1] = 0
elif TASK == 1:
    gt_seg = numpy.where(numpy.logical_or(gt_seg == task_mask[TASK][0], gt_seg == task_mask[TASK][1]))
    gt_seg[gt_seg < task_mask[TASK][0]] = 0
    gt_seg[gt_seg > task_mask[TASK][1]] = 0
    gt_seg[gt_seg == task_mask[TASK][0]] = 1
    gt_seg[gt_seg == task_mask[TASK][1]] = 1
else:
    #gt_seg = numpy.where(gt_seg == task_mask[TASK])
    gt_seg[gt_seg != task_mask[TASK]] = 0
    gt_seg[gt_seg == task_mask[TASK]] = 1

#for i in range(0, pred_seg.shape[2]):
tp = fp = fn = 0
gt2= numpy.copy(gt_seg)
pd2 = numpy.copy(pred_seg)
gt2[gt2 == 1] = 1
gt2[gt2 == 2] = 1
gt2[gt2 == 3] = 1
gt2[gt2 == 6] = 1
gt2[gt2 != 1] = 0
mm = dice_fn(pd2, gt2)
tpg += tp
fpg += fp
fng += fn
# gt2= numpy.copy(gt_seg)
# pd2 = numpy.copy(pred_seg)

# #pd2[pd2 != 1] = 0
# pd2[pd2 > 1] = 1
# mm = dice_fn(pd2, gt2)
# tpg += tp
# fpg += fp
# fng += fn
# gt2= numpy.copy(gt_seg)
# pd2 = numpy.copy(pred_seg)
# gt2[gt2 > 3] = 0
# gt2[gt2 < 2] = 0
# gt2[gt2 == 3] = 1
# gt2[gt2 == 2] = 1
# pd2[pd2 != 2] = 0
# pd2[pd2 == 2] = 1


##### EXAMPLES FIGURES
new = numpy.zeros((gt2.shape[0],gt2.shape[1],gt2.shape[2]))
new[numpy.where(numpy.logical_and(gt2 == 1, pd2 == 0))] = 3
new[numpy.where(numpy.logical_and(gt2 == 0, pd2 == 1))] = 2
new[numpy.where(numpy.logical_and(gt2 == 1, pd2 == 1))] = 1
cmap, norm = mpl.colors.from_levels_and_colors([0, 1, 2, 3, 4], ['midnightblue', 'yellow', 'red', 'green'])
cmap2, norm2 = mpl.colors.from_levels_and_colors([0, 1, 2], ['midnightblue', 'yellow'])
cmap3, norm3 = mpl.colors.from_levels_and_colors([0, 1, 2], ['midnightblue', 'orange'])
plt.figure(0)
plt.subplot(1, 3, 1)
f=plt.imshow(gt2[:, :, 235], cmap=cmap2)
f.axes.get_xaxis().set_visible(False)
f.axes.get_yaxis().set_visible(False)
plt.subplot(1, 3, 2)
f=plt.imshow(pd2[:, :, 235], cmap=cmap3)
f.axes.get_xaxis().set_visible(False)
f.axes.get_yaxis().set_visible(False)
plt.subplot(1, 3, 3)
f=plt.imshow(new[:, :, 235], cmap=cmap)
f.axes.get_xaxis().set_visible(False)
f.axes.get_yaxis().set_visible(False)
#plt.show()
plt.savefig('dod3.png', dpi=1200, bbox_inches='tight')

# mm = dice_fn(pd2, gt2)
# tpg += tp
# fpg += fp
# fng += fn
# dc= (2 * tp) / (2 * tp + fp + fn)
# dice.append(dc)
# print(str(dc * 100.0) + '%')

print("TP = " + str(tpg))
print("FP = " + str(fpg))
print("FN = " + str(fng))
print("DICE LOCAL = " + str(sum(dice) / len(dice)))
print("DICE GLOBAL = " + str((2 * tpg) / (2 * tpg + fpg + fng)))
