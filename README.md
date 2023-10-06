# Document-Level Relation Extraction with Relation Correlation Enhancement
Source code for "Document-Level Relation Extraction with Relation Correlation Enhancement", International Conference on Neural Information Processing, 2023.

## 1. Environments and Dependencies
- Python 3.9.13
- CUDA 12.0
- transformers 4.11.3
- torch 1.13.1

## 2. Dataset
- [DocRED](https://github.com/thunlp/DocRED) dataset can be downloaded following their instructions
- These files should be placed in the following format
```
 Dataset
     |- DocRED
         |- train_annotated.json        
         |- train_distant.json
         |- dev.json
         |- test.json
```
## 3. Model Training
Train LACE model using the following command, 
```bash
>> CUDA_VISIBLE_DEVICES=0 nohup sh run_bert.sh >train.log 2>&1 &
```

## 4. Result Evaluation
Experimental results are verified by transmitting the output of the test set to [CodaLab](https://competitions.codalab.org/competitions/20717#learn_the_details).

Registration is required for testing.

## 5. Citation
If you find our work inspiring, please kindly cite the following paper,

- Under ArXiv moderation, coming soon.

```bib
@
```