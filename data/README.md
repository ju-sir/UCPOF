# 数据准备说明

## 数据格式

本框架支持的数据集格式为JSON格式，每个样本包含以下结构：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "系统提示（可选）"
    },
    {
      "role": "user",
      "content": "输入句子"
    },
    {
      "role": "assistant",
      "content": "标签"
    }
  ]
}
```

## 示例数据

以下是一个ACE数据集的示例：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "I visited all their families ."
    },
    {
      "role": "assistant",
      "content": "meet"
    }
  ]
}
```

## 数据准备步骤

1. 将数据集转换为上述JSON格式
2. 在对应的数据集配置文件中（如`configs/dataset/ace.yaml`）设置正确的数据集路径
3. 确保数据集中的标签与配置文件中的`all_types`列表一致

## 数据下载

对于ACE、CASIE和AG News数据集，您可以从相应的官方网站下载原始数据，然后转换为上述格式。
