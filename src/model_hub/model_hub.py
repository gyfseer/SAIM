#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : model_hub.py
    @Time   : 2024/06/24 13:50:13
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    @Description: model hub, load model dynamically
'''

import sys
import importlib

class ModelHub:
    """
    模型仓库
    """
    
    _model_name_list = [
        "SAIM"
        ]

    _model_name_map = {
        "SAIM": {
            "module": ".SAIM.saim_model",
            "class": "SaimModel"
        }
    }

    @staticmethod
    def load_model(model_name=None):
        """
        Function:
            从模型仓库中加载所需模型

        Args:
            @param model_name: 类型 - String; 模型名称
        
        Returns:
            model: 类型 - nn.Module; 模型
        
        """
        assert model_name is not None, f"{model_name} cannot be None, select a model in {ModelHub._model_name_list}"
        
        if model_name not in ModelHub._model_name_list:
            raise ModuleNotFoundError(f"model {model_name} not found, please check in {ModelHub._model_name_list}")
        # 加载模块
        model_moule = importlib.import_module(ModelHub._model_name_map[model_name]["module"], package=__package__)
        # 从模块中获取类
        model_cls = getattr(model_moule, ModelHub._model_name_map[model_name]["class"])
        # 将类动态加载到当前模块
        setattr(sys.modules[__name__], ModelHub._model_name_map[model_name]["class"], model_cls)
        # 创建实例类
        model = model_cls()

        return model


if __name__ == "__main__":
    pass
