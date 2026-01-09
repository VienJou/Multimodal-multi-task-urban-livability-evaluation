# MMBT_liva 文件使用情况分析

## Notebook 中实际使用的文件

根据 `/u/wz53/Livability_evaluation_baseline/Livability_evaluation_baseline_EN_Clean_v5.ipynb` 的导入语句，以下文件**被使用**：

### ✅ 被使用的文件（4个）

1. **`image_liva.py`**
   - 导入：`from MMBT_liva.image_liva import ImageEncoderDenseNet`
   - 用途：图像编码器（基于 DenseNet）

2. **`mmbt_config_liva.py`**
   - 导入：`from MMBT_liva.mmbt_config_liva import MMBTConfig`
   - 用途：MMBT 模型配置类

3. **`mmbt_liva.py`**
   - 导入：`from MMBT_liva.mmbt_liva import MMBTForClassification`
   - 用途：MMBT 分类模型主类

4. **`mmbt_utils_liva_0318.py`**
   - 导入：
     - `from MMBT_liva.mmbt_utils_liva_0318 import JsonlDataset, get_image_transforms, get_labels, load_examples, collate_fn, get_multiclass_labels, get_multiclass_criterion`
     - `from MMBT_liva.mmbt_utils_liva_0318 import get_image_transforms, get_image_transforms_gray, get_image_transforms_giu`
   - 用途：数据处理工具函数（虽然 notebook 中已重写，但需要这些函数）

### ❌ 未使用的文件（18个）

这些文件是其他配置或实验的变体，**对当前 notebook 不是必需的**：

#### 不同模态配置：
- `image_liva_1img.py` - 单图像编码器
- `image_liva_2img.py` - 双图像编码器
- `mmbt_config_liva_1img.py` - 单图像配置
- `mmbt_config_liva_2img.py` - 双图像配置

#### 不同模态组合的工具：
- `mmbt_utils_liva_DSM.py` - 仅 DSM 模态
- `mmbt_utils_liva_GIU.py` - 仅 GIU 模态
- `mmbt_utils_liva_RS.py` - 仅 RS 模态
- `mmbt_utils_liva_DSM_GIU.py` - DSM + GIU
- `mmbt_utils_liva_RS_DSM.py` - RS + DSM
- `mmbt_utils_liva_RS_GIU.py` - RS + GIU

#### 单任务配置：
- `mmbt_utils_liva_singletask_phy.py` - 单任务：物理环境
- `mmbt_utils_liva_singletask_soc.py` - 单任务：社会环境
- `mmbt_utils_liva_singletask_vrz.py` - 单任务：便利性
- `mmbt_utils_liva_singletask_won.py` - 单任务：住房

#### 特殊功能版本：
- `mmbt_liva_attentionnumpy.py` - 注意力可视化版本
- `mmbt_liva_nopoi.py` - 无位置编码版本
- `mmbt_liva_XAI.py` - 可解释 AI 版本

#### 其他：
- `mmbt_utils_liva.py` - 通用工具文件（可能被 _0318 版本替代）
- `mmbt_utils_liva copy.py` - 备份文件

## 总结

- **必需文件**: 4个（`image_liva.py`, `mmbt_config_liva.py`, `mmbt_liva.py`, `mmbt_utils_liva_0318.py`）
- **可选文件**: 18个（不同实验配置的变体）
- **总文件数**: 22个

## 建议

如果只是想运行当前 notebook，可以只保留这 4 个必需文件，删除其他未使用的文件以简化代码库。但如果将来可能需要尝试不同的配置，保留所有文件也是合理的。


