import os
import pandas as pd
import dspy as dp  # 确保dspy已正确安装并支持相关功能
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir):
    """
    从指定目录加载文本数据和标签。

    Args:
        data_dir (str): 数据集目录路径，包含多个类别的子文件夹。

    Returns:
        texts (list): 文本数据列表。
        labels (list): 标签列表。
    """
    texts = []
    labels = []
    label_names = sorted(os.listdir(data_dir))  # 确保标签顺序一致
    for label in label_names:
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                file_path = os.path.join(label_dir, filename)
                try:
                    with open(file_path, 'r', encoding='latin1') as file:
                        text = file.read()
                        texts.append(text)
                        labels.append(label)
                except Exception as e:
                    logger.error(f"读取文件失败 {file_path}: {e}")
    return texts, labels

def main():
    # 定义数据集路径
    train_path = r'D:\Projects\NoobWu\DspyDemo\Datasets\20newsbydate\20news-bydate-train'
    test_path = r'D:\Projects\NoobWu\DspyDemo\Datasets\20newsbydate\20news-bydate-test'

    # 加载训练数据
    logger.info("加载训练数据...")
    X_train, y_train = load_data(train_path)
    logger.info(f"训练数据样本数：{len(X_train)}")

    # 加载测试数据
    logger.info("加载测试数据...")
    X_test, y_test = load_data(test_path)
    logger.info(f"测试数据样本数：{len(X_test)}")

    # 将数据转换为pandas DataFrame
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})

    # 假设dspy有某种方法来处理pandas DataFrame
    # 如果dspy没有直接的方法，可以使用pandas和sklearn进行后续处理
    # 以下代码使用dspy的假设方法进行演示
    try:
        logger.info("创建dspy数据集")
        train_dataset = dp.process(train_df['text'], train_df['label'])
        test_dataset = dp.process(test_df['text'], test_df['label'])
    except AttributeError:
        logger.warning("dspy没有process方法，使用pandas和sklearn进行处理")
        # 使用pandas和sklearn进行文本处理
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split

        # 文本预处理和特征提取
        logger.info("进行文本预处理和特征提取")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_train_vect = vectorizer.fit_transform(train_df['text'])
        X_test_vect = vectorizer.transform(test_df['text'])

        y_train = train_df['label']
        y_test = test_df['label']

        # 构建模型
        logger.info("训练Logistic回归模型")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_vect, y_train)

        # 预测与评估
        logger.info("进行预测")
        y_pred = model.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"模型准确率：{accuracy:.2f}")

        # 绘制混淆矩阵
        logger.info("绘制混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12,10))
        sns.heatmap(cm, annot=False, cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.show()
        return

    # 如果dspy有正确的处理流程，继续使用dspy
    # 数据处理
    logger.info("进行文本预处理")
    train_dataset = train_dataset.tokenize().remove_stopwords()
    test_dataset = test_dataset.tokenize().remove_stopwords()

    logger.info("进行特征提取")
    train_dataset = train_dataset.vectorize(method='tfidf')
    test_dataset = test_dataset.vectorize(method='tfidf')

    # 构建模型
    logger.info("训练Logistic回归模型")
    model = LogisticRegression(max_iter=1000)
    model.fit(train_dataset.features, train_dataset.labels)

    # 预测与评估
    logger.info("进行预测")
    y_pred = model.predict(test_dataset.features)
    accuracy = accuracy_score(test_dataset.labels, y_pred)
    logger.info(f"模型准确率：{accuracy:.2f}")

    # 混淆矩阵
    logger.info("绘制混淆矩阵")
    cm = confusion_matrix(test_dataset.labels, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.show()

if __name__ == "__main__":
    main()
