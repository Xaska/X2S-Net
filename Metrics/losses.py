import torch
import trimesh
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import marching_cubes
import surface_distance as surfdist
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score


def mae_3d_loss(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    mae_loss = F.l1_loss(pred, target, reduction='mean')

    return mae_loss

def hausdorff_distance(mask_pred, mask_gt):
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
    HD = surfdist.compute_robust_hausdorff(surface_distances, 95)
    return HD
def calculate_asd(mask_pred, mask_gt):
    surface_distances = surfdist.compute_surface_distances(mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
    ASD = surfdist.compute_average_surface_distance(surface_distances)
    return ASD
def surface_overlap(mask_pred, mask_gt):
    surface_distances = surfdist.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
    SO = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
    return SO
def distribution_error(mask_pred, mask_gt):
    surface_distances = surfdist.compute_surface_distances(
        mask_gt, mask_pred, spacing_mm=(1.0, 1.0, 1.0))
    Disv = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)
    return Disv
def volume_difference(mask_pred, mask_gt):
    VD = surfdist.compute_dice_coefficient(mask_gt, mask_pred)
    return VD
def accuracy_3d(predicted, truth):
    predicted_flat = predicted.view(-1)
    truth_flat = truth.view(-1)
    total_points = len(predicted_flat)
    correct_points = torch.sum(predicted_flat == truth_flat)
    accuracy = correct_points.item() / total_points
    return accuracy


def tensor_to_obj(tensor, threshold=0.5, filename='output.obj'):
    data = tensor.numpy()

    vertices, faces = marching_cubes(data, level=threshold)

    with open(filename, 'w') as f:
        for v in vertices:
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for face in faces:
            f.write('f {} {} {}\n'.format(face[0]+1, face[1]+1, face[2]+1))

