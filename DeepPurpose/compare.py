import numpy as np

# 加载两个.npy文件
v_d_muti = np.load('v_d_muti.npy')
v_d_DDI = np.load('v_d_DDI.npy')

# 比较两个数组是否完全相同
if np.array_equal(v_d_muti, v_d_DDI):
    print("两个文件的内容完全一致。")
else:
    print("两个文件的内容不一致。")

# 如果你还想看到哪里不一致（可选）
if not np.array_equal(v_d_muti[0], v_d_DDI[0]):
    # 找到第一个不同的元素的位置
    diff_idx = np.where(v_d_muti[0] != v_d_DDI[0])[0]
    if diff_idx.size > 0:
        first_diff_idx = diff_idx[0]
        print(f"第一个不同的元素位于索引：{first_diff_idx}")
        print(f"v_d_muti中的值：{v_d_muti[0][first_diff_idx]}")
        print(f"v_d_DDI中的值：{v_d_DDI[0][first_diff_idx]}")

