#cording utf-8
from datetime import datetime as time
from lib import make_dataset
from lib import cleansing
from lib import scoring
import csv

def main():
    start = time.now()
    print(start)

    #config
    #filename
    data_filename_m = './data/train.csv'
    label_filename_m = './data/label.csv'
    #param
    c = 0.5
    size = 30
    layer = 3
    lower = 4
    upper = 10
    threshold = 0.5

    '''
    プリプロセス
    '''
    # csvからデータ取得
    header, data = make_dataset.open_with_python_csv(data_filename_m)

    # 学習用labelの抽出
    label = make_dataset.label_from_csv(label_filename_m)

    #datasetの作成
    items, dataset = make_dataset.make_dataset(header, data)
    items_r, dataset_r = cleansing.cleansing(header, data, items, dataset, label)

    '''
    モデル構築プロセス
    (学習精度の確認までを実施する)
    '''
    #モデル構築
    model = scoring.logreg_modeling(dataset_r, label, c)
    model_n = scoring.neural_network_model(dataset_r, label, size, layer)
    model_g = scoring.gradientBoostingClassifier_model(dataset_r, label)

    #学習データを予測(回帰)
    score = scoring.logreg_predict(model, dataset_r, data)
    score_n = scoring.neural_network_predict(model_n, dataset_r, data)
    score_g = scoring.gradientBoostingClassifier_predict(model_g, dataset_r, data)

    #検証用label
    label_v = label

    #検証(学習精度の確認)
    scoring.accuracy(score, label_v, threshold)
    scoring.accuracy(score_n, label_v, threshold)
    scoring.accuracy(score_g, label_v, threshold)


    '''
    予測プロセスの開始
    '''
    #予測用データの作成
    data_filename_t = './data/test.csv'
    header_t, data_t = make_dataset.open_with_python_csv(data_filename_t)
    dataset_t = cleansing.remake_dataset(header, data_t, items_r)

    #予測
    score = scoring.logreg_predict(model, dataset_t, data_t)
    score_n = scoring.neural_network_predict(model_n, dataset_t, data_t)
    score_g = scoring.gradientBoostingClassifier_predict(model_g, dataset_t, data_t)

    #スコアの表示
    # for i in range(len(score)):
    #     print('logistic: %f, neural_network: %f, xgboost: %f'%(score[i][1], score_n[i][1], score_g[i][1]))

    #アンサンブル1
    rate = 0.333333
    score_f = [[score[i][0], score[i][1] * rate + score_n[i][1] * rate + score_g[i][1] * rate, label_v[i]] for i in range(len(score))]

    #アンサンブル2
    # rate = 0.5
    # score_f = [[score[i][0], score[i][1] * rate + score_n[i][1] * rate, label_v[i]] for i in range(len(score))]

    #アンサンブル3
    # rate = 0.5
    # score_f = [[score[i][0], score[i][1] * rate + score_g[i][1] * rate, label_v[i]] for i in range(len(score))]

    #アンサンブル4
    # rate = 0.5
    # score_f = [[score[i][0], score_n[i][1] * rate + score_g[i][1] * rate, label_v[i]] for i in range(len(score))]

    # only gradient boosting
    #score_f = [[score[i][0], score_g[i][1], label_v[i]] for i in range(len(score))]

    # only Neural_Network
    # score_f = [[score[i][0], score_n[i][1], label_v[i]] for i in range(len(score))]

    #提出用ファイル作成
    submit = []
    for i in range(len(score_f)):
        submit.append([score_f[i][0], 1 if score_f[i][1] >= 0.5 else 0])

    with open('./output/score.csv', 'w') as f:
         writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
         writer.writerow(['PassengerId', 'Survived'])
         writer.writerows(submit)

    end = time.now()
    print(end)
    print(end - start)

if __name__ == '__main__':
    main()
