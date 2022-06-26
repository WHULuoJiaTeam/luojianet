mindspore.dataset.text.transforms.PythonTokenizer
=================================================

.. py:class:: mindspore.dataset.text.transforms.PythonTokenizer(tokenizer)

    使用用户自定义的分词器对输入字符串进行分词。

    **参数：**

    - **tokenizer** (Callable) -  Python可调用对象，要求接收一个string参数作为输入，并返回一个包含多个string的列表作为返回值。

    **异常：**

    - **TypeError** - 参数 `tokenizer` 不是一个可调用的Python对象。
