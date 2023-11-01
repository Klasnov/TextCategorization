# Text Classification with Naive Bayes

This is a text classification project using the Naive Bayes algorithm. It classifies news headlines into different categories based on the headline text.



## Project Overview

- The project trains a Multinomial Naive Bayes model on a dataset of Chinese news headlines labeled with 14 categories. 
- The model uses Scikit-learn for implementation with CountVectorizer for text vectorization.
- Jieba is used for Chinese text segmentation. 
- The trained model is evaluated on a held-out test set, achieving 90% accuracy and strong F1 scores.



## Code Overview

- `load_dataset()` - Loads the training and test dataset from text files
- `tokenize()` - Segments the Chinese text using Jieba
- `main()` - Handles the overall workflow
  - Loads data
  - Tokenizes text
  - Vectorizes into feature matrices
  - Trains NB model
  - Makes predictions and evaluates



## Results

On the test set, the model achieves:

- Accuracy: 89.8%
- Macro F1: 87%
- Strong individual class F1 scores 

See the classification report in `main()` output for full results.



## Requirements

- Python 3.11
- Scikit-learn 1.3.2
- Jieba
- Numpy



## Usage

To run the code:

```
python text_classification.py
```



## Author

Klasnov Shao, Nov 1, 2023.

