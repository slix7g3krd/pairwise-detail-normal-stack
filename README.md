# pairwise-detail-normal-stack

## 领域

游戏、图形与多媒体

## 主题

细节法线层叠

## 说明

切线空间统一、法线 mip 重归一、距离权重衰减。

初始状态下 `graphics/detail_normal_stack.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
