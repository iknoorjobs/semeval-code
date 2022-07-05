## GateNLP-UShef at SemEval-2022 Task 8: Entity-Enriched Siamese Transformer for Multilingual News Article Similarity

Steps to train the model
1. Download the dataset available at https://competitions.codalab.org/competitions/33835#learn_the_details-timetable
2. Code to train Siamese Transformer for learning the document representations: src/train1.py (this script uses code from https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/sts/training_stsbenchmark_continue_training.py)
3. Code to extract entities from all the document pairs: src/extract.py
4. Code to train MLP using cosine scores from both Siamese Transformer and the extracted entities: src/train2.py

