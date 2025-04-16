# -*- encoding: utf-8 -*-
'''
    @文件名称   : setup-client.py
    @创建时间   : 2023/12/27 16:09:44
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 客户端启动文件
    @参考地址   : 无
'''
import logging
from src.client import main

if __name__ == '__main__':
    # config logging
    logging.basicConfig(filename=f'./results/log/client_log.txt',
                     format = '%(asctime)s - %(levelname)s - %(message)s - %(funcName)s',
                     level=logging.INFO)
    main()