import math
import numpy as np
import torch
import scipy
# def cal_ef(midpoint, midpoint_s, large_trace, small_trace):
#     num_parts = 20
#     part_length = midpoint.item() / num_parts
#     part_length_s = midpoint_s.item() / num_parts
#     # 计算每个点到两边边缘的直线距离
#     mask_distances = []
#     mask_distances_s = []
#     # distance = scipy.ndimage.distance_transform_edt(large_trace, sampling=[1, 1])
#     # 找到掩码区域的索引位置
#     mask_indices = torch.nonzero(large_trace)
#     mask_indices_s = torch.nonzero(small_trace)
#     # 找到掩码区域的最左边和最右边的像素位置
#     left_edge = torch.min(mask_indices[:, 2])
#     right_edge = torch.max(mask_indices[:, 2])
#
#     left_edge_s = torch.min(mask_indices_s[:, 2])
#     right_edge_s = torch.max(mask_indices_s[:, 2])
#     # 计算每个点到两边边缘的直线距离
#     for i in range(num_parts):
#         # 计算当前部分的边缘位置
#         part_left_edge = left_edge + i * part_length
#         part_right_edge = right_edge + i *part_length
#         part_left_edge_s = left_edge_s + i * part_length_s
#         part_right_edge_s = left_edge_s + i* part_length_s
#
#         # 计算当前部分的掩码距离
#         part_mask = mask_indices[(mask_indices[:, 2] >= part_left_edge) & (mask_indices[:, 2] < part_right_edge)]
#         part_distance = torch.max(part_mask[:, 2]) - torch.min(part_mask[:, 2])
#         mask_distances.append(part_distance)
#         part_mask_s = mask_indices_s[(mask_indices_s[:, 2] >= part_left_edge_s) & (mask_indices_s[:, 2] < part_right_edge_s)]
#         part_distance_s = torch.max(part_mask_s[:, 2]) - torch.min(part_mask_s[:, 2])
#         mask_distances_s.append(part_distance_s)
#     EDV=np.pi/4*midpoint/20*sum(mask_distances)
#     ESV = np.pi / 4 * midpoint_s / 20 * sum(mask_distances_s)
#     EF=(EDV-ESV)/EDV
#     return EDV,ESV,EF
def cal_ef(midpoint, midpoint_s, large_trace, small_trace):
    EDV=midpoint*large_trace
    ESV=midpoint_s*small_trace
    EF=(EDV-ESV) /EDV

    return EDV,ESV,EF


def cal_edv(distance_1_to_midpoint,large_trace):
    num_parts = 20
    part_length = distance_1_to_midpoint.item() / num_parts
    # 计算每个点到两边边缘的直线距离
    mask_distances = []
    # 找到掩码区域的索引位置
    mask_indices = torch.nonzero(large_trace)
    # 找到掩码区域的最左边和最右边的像素位置
    left_edge = torch.min(mask_indices[:, 2])
    right_edge = torch.max(mask_indices[:, 2])

    # 计算每个平分点到两边边缘的直线距离
    for i in range(1, num_parts + 1):
        # 计算当前平分点的位置
        part_point = i * part_length
        # 计算当前平分点到两边边缘的直线距离
        left_distance = part_point - left_edge
        right_distance = right_edge - part_point
        part_distance = left_distance + right_distance
        cul_distance = part_distance * part_distance
        mask_distances.append(part_distance)
    EDV = np.pi/4*distance_1_to_midpoint/20*sum(cul_distance)
    return EDV

def cal_esv(distance_1_to_midpoint_s, small_trace):
    num_parts = 20
    part_length = distance_1_to_midpoint_s.item() / num_parts
    # 计算每个点到两边边缘的直线距离
    mask_distances = []
    # 找到掩码区域的索引位置
    mask_indices = torch.nonzero(small_trace)
    # 找到掩码区域的最左边和最右边的像素位置
    left_edge = torch.min(mask_indices[:, 2])
    right_edge = torch.max(mask_indices[:, 2])

    # 计算每个平分点到两边边缘的直线距离
    for i in range(1, num_parts + 1):
        # 计算当前平分点的位置
        part_point = i * part_length
        # 计算当前平分点到两边边缘的直线距离
        left_distance = part_point - left_edge
        right_distance = right_edge - part_point
        part_distance = left_distance + right_distance
        cul_distance = part_distance * part_distance
        mask_distances.append(part_distance)
    ESV = np.pi/4*distance_1_to_midpoint_s/20*sum(cul_distance)
    return ESV