# flake8: noqa
from .data.DocVQA import DocVQADataset
from .modeling.layoutlm import (
    LayoutlmConfig,
    LayoutlmForSequenceClassification,
    LayoutlmForTokenClassification,
)
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
