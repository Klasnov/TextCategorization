"""
基于朴素贝叶斯的文本分类

本项目使用朴素贝叶斯算法进行文本分类实验。本次实验为新闻标题文本分类，根据给出的新闻标
题文本和标签训练一个分类模型，然后对测试集的新闻标题文本进行分类。

作者：邵彦铭
时间：2023年11月1日
环境：Python 3.11，scikit-learn 1.3.2
"""

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# 加载数据集
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split('\t')
            dataset.append((text, label))
    return dataset

# 文本数据分词
def tokenize(text):
    return ' '.join(jieba.cut(text))

# 主函数
def main():
    # 导入实验数据
    train_data = load_dataset('data/train.txt')
    test_data = load_dataset('data/test.txt')

    # 分词
    train_data = [(tokenize(text), label) for text, label in train_data]
    test_data = [(tokenize(text), label) for text, label in test_data]

    # 文本向量化
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([text for text, _ in train_data])
    y_train = [label for _, label in train_data]

    X_test = vectorizer.transform([text for text, _ in test_data])
    y_test = [label for _, label in test_data]

    # 构建与训练朴素贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # 进行模型性能测试
    y_test_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    print()
    print("Test Accuracy:", accuracy)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
