# encoding: utf-8
"""
@author:    dawei gao
@contact:   david_gao@buaa.edu.cn

@version: 1.0
@license: Apache Licence
@file: icml07.py
@time: 2018/10/20 7:24 PM

"""

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import datetime

pd_list = []
pf_list = []
g_mean_list = []


def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    # print('혼동행렬 : ', cm)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    # print('PD : ', PD)
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    # print('PF : ', PF)
    G_Mean = geometric_mean_score(y_test, y_pred)
    # print('G-mean : ', G_Mean)

    return PD, PF, G_Mean


def trainWeightedClassifier(data_training, labels_training, weights):
    model = svm.LinearSVC(verbose=0, max_iter=5000)
    # model = tree.DecisionTreeClassifier(criterion="gini", max_features="log2", splitter="random")
    model.fit(data_training, labels_training, sample_weight=weights)
    return model


def loadData(path):
    file = open(path)
    labels = list()
    data = list()
    for line in file:
        labels = labels + [int(line.split('\t')[0])]
        data = data + [[int(i) for i in line.split('\t')[1:]]]
    return np.array(data), np.array(labels)


def predict_final(model_list, beta_t_list, data, label, threshold):
    res = 1
    for i in range(len(model_list)):
        h_t = model_list[i].predict([data])[0]
        res = res / beta_t_list[i] ** h_t
    if res >= threshold:
        label_predict = 1
    else:
        label_predict = 0
    if label_predict == label:
        return 1
    else:
        return 0


def error_calculate(model, training_data_target, training_labels_target, weights):
    total = np.sum(weights)
    labels_predict = model.predict(training_data_target)
    error = np.sum(weights / total * np.abs(labels_predict - training_labels_target))
    return error


def TrAdaBoost(N=100):
    # 데이터 처리
    training_data_source, training_labels_source = loadData('./datasets/mushroom_tapering')
    data_target, labels_target = loadData('./datasets/mushroom_enlarging')

    training_data_target, test_data_target, training_labels_target, test_labels_target = train_test_split(data_target,
                                                                                                          labels_target,
                                                                                                          test_size=0.25)

    # 합성 훈련 데이터
    training_data = np.r_[training_data_source, training_data_target]
    training_labels = np.r_[training_labels_source, training_labels_target]

    # 비교 시험 baseline 방법
    svm_0 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_0.fit(training_data, training_labels)
    print('——————————————————————————————————————————————')
    print('트레이닝 데이터용 타겟 도메인 및 소스 도메인의 경우')
    print('The mean accuracy is ' + str(svm_0.score(test_data_target, test_labels_target)))
    print('The error rate is ' + str(1 - svm_0.score(test_data_target, test_labels_target)))

    svm_1 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_1.fit(training_data_target, training_labels_target)
    print('——————————————————————————————————————————————')
    print('훈련 데이터가 대상 도메인만 사용되는 경우')
    print('The mean accuracy is ' + str(svm_1.score(test_data_target, test_labels_target)))
    print('The error rate is ' + str(1 - svm_1.score(test_data_target, test_labels_target)))

    svm_2 = svm.LinearSVC(verbose=0, max_iter=5000)
    svm_2.fit(training_data_source, training_labels_source)
    print('——————————————————————————————————————————————')
    print('훈련 데이터가 소스 도메인만 사용되는 경우')
    print('The mean accuracy is ' + str(svm_2.score(test_data_target, test_labels_target)))
    print('The error rate is ' + str(1 - svm_2.score(test_data_target, test_labels_target)))
    print('——————————————————————————————————————————————')

    # 훈련 메인 사이클
    n_source = len(training_data_source)
    m_target = len(training_data_target)
    # 초기화 가중치
    weights = np.concatenate((np.ones(n_source) / n_source, np.ones(m_target) / m_target))
    beta_t_list = list()
    model_list = list()
    beta = 1.0 / (1.0 + np.sqrt(2 * np.log(n_source) / N))
    # N=100
    for t in range(N):
        p_t = weights / sum(weights)
        model = trainWeightedClassifier(training_data, training_labels, p_t)

        # 가중 오류율
        error_self = error_calculate(model, training_data_target, training_labels_target, weights[-m_target:])

        # 계산 파라미터
        if error_self > 0.5:
            error_self = 0.5
        elif error_self == 0:
            t = N
            break

        beta_t = error_self / (1 - error_self)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '제' + str(t) + '바퀴의 가중 오류율: ' + str(
            error_self))

        # 소스 도메인
        for i in range(n_source):
            if model.predict([training_data_source[i]])[0] != training_labels_source[i]:
                weights[i] = weights[i] * beta

        # 대상 도메인
        for i in range(m_target):
            if model.predict([training_data_target[i]])[0] != training_labels_target[i]:
                weights[i + n_source] = weights[i + n_source] / beta_t

        # 현재 인자 기록하기
        beta_t_list += [beta_t]
        model_list += [model]

    # 최종 출력 모델을 테스트합니다.
    count_accu = 0
    index_half = int(np.ceil(N / 2))
    threshold = 1
    index_half = int(index_half)
    for beta_t in beta_t_list[index_half:]:
        threshold = threshold / np.sqrt(beta_t)

    for i in range(len(test_data_target)):
        count_accu += predict_final(model_list[index_half:], beta_t_list[index_half:], test_data_target[i],
                                    test_labels_target[i], threshold)
    error_final = 1.0 - count_accu / float(len(test_data_target))
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '모델의 최종 정확도 : ' + str(1 - error_final))
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + '모델의 최종 오류율 : ' + str(error_final))


if __name__ == '__main__':
    TrAdaBoost()

