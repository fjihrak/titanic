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
    data_filename_p = './data/train.csv'
    label_filename_p = './data/label.csv'
    #param
    c = 0.5
    size = 100
    layer = 100
    lower = 4
    upper = 10
    threshold = 0.5

    # csvからデータ取得
    header, data = make_dataset.open_with_python_csv(data_filename_m)

    # 学習用labelの抽出
    label = make_dataset.label_from_csv(label_filename_m)

    #datasetの作成
    items, dataset = make_dataset.make_dataset(header, data)

    items_r, dataset_r = cleansing.cleansing(header, data, items, dataset, label)

    #モデル構築
    model = scoring.logreg_modeling(dataset_r, label, c)
    model_n = scoring.neural_network_model(dataset_r, label, size, layer)
    model_g = scoring.gradientBoostingClassifier_model(dataset_r, label)

    #予測用データの作成
    header_p, data_p = make_dataset.open_with_python_csv(data_filename_p)
    dataset_p = cleansing.remake_dataset(header, data_p, items_r)

    #予測
    # score = scoring.logreg_predict(model, dataset_p, data_p)
    # score_n = scoring.neural_network_predict(model_n, dataset_p, data_p)
    # score_g = scoring.gradientBoostingClassifier_predict(model_g, dataset_p, data_p)

    #検証用データの作成
    label_v = make_dataset.label_from_csv(label_filename_p)

    #検証
    score = scoring.logreg_validation(model, dataset_p, data_p, label_v)
    score_n = scoring.neural_network_validation(model_n, dataset_p, data_p, label_v)
    score_g = scoring.gradientBoostingClassifier_validation(model_g, dataset_p, data_p, label_v)

    scoring.accuracy(score, label_v, threshold)
    scoring.accuracy(score_n, label_v, threshold)
    scoring.accuracy(score_g, label_v, threshold)

    #アンサンブル
    score_f = []
    rate = 0.5
    for i in range(len(score)):
        #score_f.append([score[i][0], score_g[i][1] * rate + score_n[i][1] * (1 - rate)])
        score_f.append([score[i][0], score[i][1] * rate + score_n[i][1] * (1 - rate), label_v[i]])
    scoring.accuracy(score_f, label_v, threshold)

    #提出用ファイル作成
    submit = []
    for i in range(len(score_f)):
        submit.append([score_f[i][0], 1 if score_g[i][1] >= 0.5 else 0])

    with open('./output/score.csv', 'w') as f:
         writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
         writer.writerow(['PassengerId', 'Survived'])
         writer.writerows(submit)

    end = time.now()
    print(end)
    print(end - start)

if __name__ == '__main__':
    main()
