import gc
import logging
import os

import numpy as np
import pandas as pd
import torch

import data_preprocess
from args import args
from dataset import load_data
from roberta_model import load_model
from running import Running
from torchcontrib.optim import SWA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    # 训练epochs轮
    for epoch in range(args.epochs):
        # 载入模型
        tokenizer, model = load_model(args)
        # 读取数据
        data_path = args.data_path + str(epoch)
        train_data_path = os.path.join(data_path, 'train.csv')
        train_loader = load_data(tokenizer, args, train_data_path, "train")
        evaluate_data_path = os.path.join(data_path, 'dev.csv')
        evaluate_loader = load_data(tokenizer, args, evaluate_data_path, "evaluate")
        logger.info("Training data has been loaded!")
        # 训练
        running = Running(model, args)
        model = running.model
        # 定义优化器
        base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup,
                                                    num_training_steps=(args.epochs * len(
                                                        train_loader) / args.batch_accumulation))

        # 训练
        running.train(train_loader, evaluate_loader, optimizer, scheduler, epoch)

        torch.cuda.empty_cache()  # 释放显存
        gc.collect()  # 垃圾回收
        # 模型的权重设置为它们的SWA平均值
    optimizer.swap_swa_sgd()

def test():
    for epoch in range(args.epochs):
        # 载入模型
        output_path = args.output_path + str(epoch)
        args.model_path = os.path.join(output_path, "pytorch_model.bin")
        tokenizer, model = load_model(args)
        # 载入测试集数据
        data_path = args.data_path + str(epoch)
        test_data_path = os.path.join(data_path, 'test.csv')
        test_loader = load_data(tokenizer, args, test_data_path, "test")  # 3263
        logger.info("Testing data has been loaded!")
        # 得到测试结果
        running = Running(model, args)
        outputs = running.test(test_loader)
        # 写入数据
        outputs_df = pd.read_csv(os.path.join(args.raw_data_path, "sample_submission.csv"))
        outputs_df['target_0'] = outputs[:, 0]
        outputs_df['target_1'] = outputs[:, 1]
        outputs_df['target_2'] = outputs[:, 2]
        outputs_df[['id', 'target_0', 'target_1', 'target_2']].to_csv(os.path.join(output_path, "sub.csv"), index=False)
        logger.info('sub ' + str(epoch) + ' has been written.')


# 生成最终结果
def generate_result():
    submit_df = pd.read_csv(os.path.join(args.raw_data_path, "sample_submission.csv"))
    submit_df['0'] = 0
    submit_df['1'] = 0
    submit_df['2'] = 0
    for epoch in range(0, args.epochs):
        output_path = args.output_path + str(epoch)
        tmp = pd.read_csv(os.path.join(output_path, 'sub.csv'))
        submit_df['0'] += tmp['target_0'] / args.epochs
        submit_df['1'] += tmp['target_1'] / args.epochs
        submit_df['2'] += tmp['target_2'] / args.epochs
    submit_df['id'] = submit_df['id'].astype(str) + ' '
    submit_df['y'] = np.argmax(submit_df[['0', '1', '2']].values, -1) - 1
    submit_df[['id', 'y']].to_csv(os.path.join(args.submit_path, 'submit.csv'), index=False)
    logger.info("The final result has been generated.")


if __name__ == '__main__':
    # 切分训练集和验证集数据
    data_preprocess.cut_fold(k=args.epochs)
    # 训练
    train()
    test()
    generate_result()
