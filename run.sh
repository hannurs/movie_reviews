#!/bin/sh
python build_vocabulary.py
python extract_features.py
python train_classifier.py
python validate.py