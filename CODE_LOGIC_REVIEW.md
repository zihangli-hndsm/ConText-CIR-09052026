# ConText-CIR 代码逻辑审查报告

> 审查范围：`src/` 下训练、数据、评测与提交脚本。侧重“缺失模块”和“算法/实现错误”。

## 1) 缺失模块 / API 不一致（会直接导致运行失败）

### 1.1 评测与提交脚本引用了不存在的数据集类
- `src/eval.py` 导入了 `GalleryDataset`、`CIRRDataset`，但 `src/Dataset.py` 仅定义了 `CIRDataset` 与 `collate_fn_with_nps`。
- `src/circo_sub.py` 导入了 `CIRRDataset`、`CIRCODataset`，同样在 `src/Dataset.py` 中不存在。

**影响**：脚本启动即 `ImportError`，评测/提交流程不可用。

### 1.2 `ConText` 初始化接口与加载方式存在潜在不兼容
- `ConText.__init__` 强制要求 `args` 参数（且依赖 `args.backbone_size`、`args.learning_rate`）。
- `eval.py` / `circo_sub.py` 使用 `ConText.load_from_checkpoint(path)` 时并未显式提供 `args`。

**影响**：若 checkpoint 中未完整保存并可恢复 `args`，则会在恢复模型时失败（不同 Lightning 版本/保存策略下较常见）。

## 2) 数据管线中的关键逻辑错误

### 2.1 测试集判断条件写法错误（字符串包含而非相等比较）
`src/Dataset.py` 中：
- `tgt_filename = None if split in 'test1' and dataset == 'cirr' else ...`

这里 `split in 'test1'` 是“字符包含”语义：
- `split='test'` 时也会为真（因为 `'test'` 是 `'test1'` 子串）；
- 任何包含 `t/e/s/t/1` 组合的异常值也可能误命中。

**正确意图**应为 `split == 'test1'`。

**影响**：目标图像路径构造逻辑可能在非预期 split 被跳过，导致训练/验证样本标签丢失或行为不一致。

### 2.2 文件名处理使用 `strip('.png')` 存在算法语义错误
多个位置用到了：`xxx.split('/')[-1].strip('.png')`

`strip('.png')` 不是“去后缀”，而是“去掉两端任意属于 `. p n g` 的字符”。
例如：
- `"spring.png" -> "spri"`（尾部 `ng` 也会被连续剥离）
- `".png_car.png"` 结果不可预期。

**正确做法**应使用 `Path(name).stem` 或 `os.path.splitext(name)[0]`。

**影响**：图像 key 与 `mapper` 对不上，进而触发 `KeyError` 或错误映射到其他图片。

### 2.3 `collate_fn_with_nps` 对 `target` 无条件 `torch.stack`，与测试样本 `target=None` 冲突
- `__getitem__` 在某些 split（尤其测试）会返回 `target=None`。
- `collate_fn_with_nps` 执行 `torch.stack([b['target'] for b in batch])`。

**影响**：测试/无标注场景下会直接报错，导致 DataLoader 迭代失败。

## 3) 训练与评测逻辑中的高风险问题

### 3.1 `Train.py` 中的 `create_optimized_transforms(use_channel_last=True)` 很可能类型错误
- 在 `ToTensor()` 后追加 `Lambda(lambda x: x.to(memory_format=torch.channels_last))`。
- `channels_last` 适用于 4D 张量（NCHW）内存格式；单张图像张量为 3D（CHW）。

**影响**：开启 `--use_channel_last` 时可能在预处理阶段触发运行时错误，训练无法开始。

### 3.2 `Parser` 在模型初始化时直接构建 Stanza pipeline，缺少资源与环境兜底
- `Parser.__init__` 直接 `stanza.Pipeline(...)`。
- 若环境未预下载 stanza 模型包，会在启动训练时失败（尤其离线/新环境）。

**影响**：模型构造阶段即中断；与 README 的“安装后可直接训练”预期不一致。

### 3.3 名词短语抽取“span”是词级 span，但后续常用于 token 级特征监督，语义可能错位
- `Parser` 基于 constituency tree 的 `leaves()` 位置生成 span（词级）。
- CLIP tokenizer 是子词粒度（BPE），若后续用这些 span 去对齐 token embedding，需要明确词到子词映射。

**影响**：若未做词-子词对齐，概念一致性损失可能对错位置监督，造成训练信号噪声（性能不稳定/上限受限）。

### 3.4 四层 Cross-Attention 堆叠缺少残差连接
- `ConText.forward` 在循环中执行 `vision_features = cross_attn(vision_features, text_features)`，每层都直接覆盖上一层表示。
- 这与常见 Transformer block 的“Attention + Residual”设计不一致；当前结构会丢失前一层原始信息通路。

**影响**：深层堆叠时梯度与特征传递更脆弱，训练稳定性和检索效果可能受影响，尤其在 cross-attention 层数增加时更明显。

## 4) 工程可维护性问题（非致命，但建议尽快修）

1. `src/utils.py` 中调用 NVML 后未看到 `nvmlShutdown()`，长进程/重复调用可能造成资源泄漏风险。  
2. `README.md` 的运行命令路径（`python Train.py`）与实际文件位于 `src/Train.py` 不一致，易误导新用户。  
3. `eval.py` / `circo_sub.py` 顶层解析参数并执行逻辑，模块复用性差，且不便单测。  

## 5) 修复优先级建议

### P0（先修，不修就跑不起来）
- 补全/统一 `Dataset` 中被评测脚本依赖的数据集类（或修改脚本统一使用 `CIRDataset`）。
- 修复 `split in 'test1'` 为严格相等判断。
- 修复 `strip('.png')` 为标准去后缀。
- 修复 `collate_fn_with_nps` 对 `target=None` 的兼容。

### P1（能跑但容易炸）
- 处理 `load_from_checkpoint` 与 `args` 的兼容策略（checkpoint 超参回填、默认配置对象、或 classmethod 包装）。
- 修复 `use_channel_last` 的 3D/4D 使用错误。
- 为 stanza 资源初始化增加检查与提示（必要时惰性加载）。
- 为每层 cross-attention 增加残差路径（必要时补 LayerNorm/FFN 形成完整 block）。

### P2（提升稳定性和可复现性）
- 明确并实现词级 span 到 CLIP 子词 span 的对齐策略。
- 调整 README 与代码结构一致性，补充最小可运行 eval/submission 示例。
