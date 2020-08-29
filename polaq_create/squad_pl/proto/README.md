The real `CoreNLP_pb2.py` generated protobuf file is in `stanfordnlp` package
so you shouldn't compile `proto/CoreNLP.proto`.

If you want to compile provided `dataset.proto` then:

1\. Compile `dataset.proto` with `protoc`:

```shell script
protoc --python_out=. dataset.proto
```

2\. In generated `dataset_pb2.py` file change line (at the beginning of the file)

```python
import CoreNLP_pb2 as CoreNLP__pb2
```
to
```python
from stanfordnlp.protobuf import CoreNLP_pb2 as CoreNLP__pb2
```
