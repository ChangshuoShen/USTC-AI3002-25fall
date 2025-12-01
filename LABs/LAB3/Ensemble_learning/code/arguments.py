import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type=str, default='all',
                    choices=['voting', 'bagging', 'adaboost', 'gbdt', 'stacking', 'all'],
                    help='选择要运行的集成学习方法')

    parser.add_argument('--n_estimators', type=int, default=50,
                        help='弱学习器数量')

    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='AdaBoost 学习率')

    parser.add_argument('--save_dir', type=str, default='../results/',
                        help='结果保存目录')

    return parser.parse_args()
