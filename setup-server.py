#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : setup-server.py
    @Time   : 2024/06/20 14:35:09
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    @Description: write here
'''

import logging
import argparse

from src.server import Server
from src.global_var import GlobalVar


def main(args):
    server = Server(args)
    server.run()


if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(filename=f'./results/log/server_log.txt',
                        format='%(asctime)s - %(levelname)s - %(message)s - %(funcName)s',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    # 指定模型
    parser.add_argument(
        "--model_name",
        default="SAIM",
        metavar="string",
        help="model name",
    )
    args = parser.parse_args()

    GlobalVar.current_model_name = args.model_name
    main(args)
