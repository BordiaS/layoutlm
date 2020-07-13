#!/bin/bash
#wget https://guillaumejaume.github.io/FUNSD/dataset.zip

#unzip dataset.zip && mv dataset data && rm -rf dataset.zip __MACOSX

python preprocess.py --data_dir /datatop_2/Shikha/docie/DocVQA/val/ocr_results \
                                    --data_split val \
                                    --output_dir /datatop_2/Shikha/docie/DocVQA \
                                    --model_name_or_path bert-base-uncased \
                                    --max_len 510

python preprocess.py --data_dir /datatop_2/Shikha/docie/DocVQA/test/ocr_results \
                                    --data_split test \
                                    --output_dir /datatop_2/Shikha/docie/DocVQA \
                                    --model_name_or_path bert-base-uncased \
                                    --max_len 510


#python preprocess.py --data_dir /datatop_2/Shikha/docie/FunSD/testing_data/annotations \
 #                                   --data_split test \
  #                                  --output_dir /datatop_2/Shikha/docie/FunSD \
   #                                 --model_name_or_path bert-base-uncased \
    #                                --max_len 510

#cat /datatop_2/Shikha/docie/FunSD/train.txt | cut -d$'\t' -f 2 | grep -v "^$"| sort | uniq > /datatop_2/Shikha/docie/FunSD/labels.txt