# 中文纠错 



## softmaskbert

> 模型下载后放到gec_ch/models/softmaskbert文件夹下

### 调用方式

```python
 from gec_zh.softmaskbert.correct import correct_text
 print(correct_text("你叫什么名子，哪所学笑呢？"))
```

### 返回格式

```json
 {'original_text': '你叫什么名子，哪所学笑呢？',
  'corrected_text': '你叫什么名字，哪所学校呢？',
  'details': [{'pos': 5, 'ori': '子', 'cor': '字'}, {'pos': 10, 'ori': '笑', 'cor': '校'}]
 }
```

- original_text:原文
- corrected_text:修改之后的文本
- details: 修改细节
- pos: 修改位置
- ori: 修改前字符
- cor: 修改后的字符

