# BI-LSTM-CRF-for-Aspect-Extraction-Sentiment-Extraction

data   ->   Dataset modified by training data for BDCI 2017 Topic Based Text Sentiment Analysis: http://www.datafountain.cn/#/competitions/268/intro, last accessed 2018/5/13

train.csv -> training file size:13652
dev.csv -> development file size:2000
test.csv -> test file size:2000

pre_data.py   ->    generate a dictinoary for random embedding and label2tag

model.py    ->     implemention for BI-LSTM-CRF/BI-LSTM/LSTM-CRF/LSTM-CRF(swtich by flag)

main.py    ->      main file

conlleval_rev.pl    ->    evaluation manuscript for SINHAN NER task

conlleval.py  ->   evaluation metrics for this task, which can be used for sequence labeling task
