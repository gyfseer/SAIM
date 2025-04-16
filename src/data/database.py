# -*- encoding: utf-8 -*-
'''
    @文件名称   : database.py
    @创建时间   : 2023/12/28 15:07:31
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 数据库管理，数据库的连接以及数据库内容的增删改查接口管理。
    @参考地址   : 无
'''

import json
import shutil
import getpass
import pymysql
import argparse
import traceback

from tqdm import tqdm

IMAGE_ROOT_DIR = "datasets/train/"

class MySQLDatabase:
        
    db = None

    def __init__(self, password, host="localhost", port=3306, user="root", database_name="incre_database"):
        # 创建链接
        self.connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        
        # 创建游标
        self.cursor = self.connection.cursor()

        # 指明数据库
        self.cursor.execute(f"USE {database_name}")

    def get_image_root_dir(self):
        """
        图片根路径, 程序初始化时指定, 以系统分隔符结尾
        """
        return IMAGE_ROOT_DIR
    
    def load_annotations(self, jsonfile, image_origin_dir):
        """
        将数据集标注文件加载到数据库
        1. 将标注信息加载到数据库
        2. 将图片从原位置复制到指定位置

        Args:
            @Param jsonfile (str): 要加载的任务标注信息所在路径
            @Param image_origin_dir (str): 图片原位置, 即要加载到数据库的标签对应的图片的目录
        """
        print(f"接下来的操作将加载：{jsonfile}内的数据到数据库")
        with open(jsonfile, "r")  as fp:
            annotations = json.load(fp)
        # 开启事务
        self.connection.begin()
        self.cursor.execute(f"DESCRIBE t_images_categories_map")
        num_columns = len(self.cursor.fetchall())
        placeholder = (num_columns-2) * f" 1.0,"
        placeholder = placeholder[:-1]
        try:
            # 将类别信息加载到数据库
            categories_table =  "t_categories"
            sql = f"SELECT `name` FROM {categories_table}"
            self.cursor.execute(sql)
            results = [x[0] for x in self.cursor.fetchall()]
            categories_set = set(results)
            sql = f"SELECT max(`id`) FROM {categories_table}"
            self.cursor.execute(sql)
            try:
                category_id = self.cursor.fetchall()[0][0] + 1
            except Exception:
                category_id = 1

            # category_id = len(results) + 1
            for category in annotations["categories"]:
                if category["name"] in categories_set:
                    continue
                sql = f"INSERT INTO {categories_table}(id, name) VALUES(%s, %s)"
                values = (category_id, category["name"])
                self.cursor.execute(sql, values)
                category_id += 1
            
            # 建立名称和类别的映射
            dataset_categories_map = { category["id"]:category["name"] for category in annotations["categories"] }
            sql = f"SELECT * FROM {categories_table}"
            self.cursor.execute(sql)
            database_categories_map = { result[1]:int(result[0]) for result in self.cursor.fetchall()}

            # 插入图片数据
            image_table = "t_images"
            image_categories_map_table = "t_images_categories_map"
            sql = f"SELECT max(`id`) FROM {image_table}"
            self.cursor.execute(sql)
            try:
                database_image_id = self.cursor.fetchall()[0][0] + 1
            except Exception:
                database_image_id = 0
            # 批量插入数据的大小
            BATCH_SIZE = 1000

            # 初始化批量插入的数据列表
            batch_insert_values = []
            batch_insert_categories = []
            image_id_map = {}
            for anno_info in annotations["annotations"]:
                if anno_info["image_id"] in image_id_map:
                    image_id_map[anno_info["image_id"]]["bbox"].append(anno_info["bbox"])
                    image_id_map[anno_info["image_id"]]["label"].append(database_categories_map[dataset_categories_map[anno_info["category_id"]]])
                else:
                    image_id_map[anno_info["image_id"]] = {
                        "bbox": [anno_info["bbox"]],
                        "label":[database_categories_map[dataset_categories_map[anno_info["category_id"]]]]
                    }
            count = 0
            for image_info in tqdm(annotations["images"]):
                count += 1
                image_id = image_info["id"]
                if image_id not in image_id_map:
                    continue
                # 构建第一个插入语句的数据
                insert_values = (database_image_id, json.dumps({"bbox": image_id_map[image_id]["bbox"], "label": image_id_map[image_id]["label"]}))
                batch_insert_values.append(insert_values)
                # 构建第二个插入语句的数据
                for category_id in set(image_id_map[image_id]["label"]):
                    insert_categories = (database_image_id, category_id)
                    batch_insert_categories.append(insert_categories)

                image_origin_path = image_origin_dir + image_info["file_name"]
                image_target_path = IMAGE_ROOT_DIR + str(database_image_id) + ".jpg"
                # shutil.copyfile(image_origin_path, image_target_path)
                shutil.copyfile(image_origin_path, image_target_path)
                database_image_id += 1
                # 当批量插入数据达到 BATCH_SIZE 或处理完所有数据时，执行数据库插入操作
                if len(batch_insert_categories) >= BATCH_SIZE or count == len(annotations["images"]):
                    # 执行第二个插入语句
                    sql = f"INSERT INTO {image_categories_map_table} VALUES (%s, %s, {placeholder})"
                    self.cursor.executemany(sql, batch_insert_categories)
                    # 执行第一个插入语句
                    sql = f"INSERT INTO {image_table}(id, annotations) VALUES (%s, %s)"
                    self.cursor.executemany(sql, batch_insert_values)

                    # 清空批量插入的数据列表
                    batch_insert_values = []
                    batch_insert_categories = []

            self.connection.commit()
        except Exception as e:
            traceback.print_exc()
            self.connection.rollback()
    
    def check_score_column(self, model_name, default_score=1.0):
        """
            检查model_name对应的列是否存在, 不存在插入新的列 _scores
        """
        sql = f"SHOW COLUMNS FROM t_images_categories_map LIKE '{model_name}_scores'"
        self.cursor.execute(sql)
        if not self.cursor.fetchone():
            self.connection.begin()
            try:
                sql = f"ALTER TABLE t_images_categories_map ADD COLUMN {model_name}_scores FLOAT DEFAULT {default_score}"
                self.cursor.execute(sql)
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise AssertionError("add column failed")

    def select_image_all(self):
        sql = f"SELECT * FROM t_images"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        results = [ (res[0], json.loads(res[1])) for res in results]
        return results

    def select_category_names(self):
        """
            查询已有类别的种类名称

        Returns :
            种类名称列表
        """
        sql = f"SELECT `name` FROM t_categories"
        self.cursor.execute(sql)
        category_list = [ x[0] for x in self.cursor.fetchall()]
        print(category_list)
        
        return category_list
    
    def select_category_ids(self):
        """
            查询`t_categories`中的所有id
        """
        categories_table =  "t_categories"
        sql = f"SELECT `id` FROM {categories_table}"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        return [x[0] for x in results ]

    def select_category_map(self):
        """
            查询`t_categories`中的id与name的映射
        """
        categories_table =  "t_categories"
        sql = f"SELECT * FROM {categories_table}"
        self.cursor.execute(sql)
        results = { r[0]:r[1] for r in self.cursor.fetchall()}
        return results

    def select_image_id_prob_by_category_id(self, category_id):
        """
            通过类别id查询图片id 和 图片的采样概率

        """
        image_categories_map_table = "t_images_categories_map"
        sql = f"SELECT * FROM {image_categories_map_table} WHERE category_id={category_id}"
        self.cursor.execute(sql)
        image_ids = []
        scores = []
        for res in self.cursor.fetchall():
            image_ids.append(res[0])
            scores.append(res[2])
        return image_ids, scores

    def select_image_by_id(self, image_ids):
        """
                根据图片id查询图片的标准信息
        """
        image_table = "t_images"
        sql = f"SELECT * from {image_table} WHERE id = %s"
        results = []
        for id in image_ids:
            self.cursor.execute(sql, id)
            results.append(self.cursor.fetchall()[0])
        return results
    
    def select_imageid_and_scores_by_modelname(self, model_name):
        sql = f"select image_id, {model_name}_scores from t_images_categories_map"
        self.cursor.execute(sql)
        scores_info = self.cursor.fetchall()
        return scores_info[0]

    def update_scores_by_modelname(self, new_scores, modelname):
        """
            批量更新图片的采样得分信息.
        """
        self.connection.begin()
        try:
            sql = f"UPDATE `t_images_categories_map` SET {modelname}_scores=%s WHERE image_id = %s"
            self.cursor.executemany(sql, new_scores)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
    
    def upate_score_by_category_id(self, category, number, model_name):
        self.connection.begin()
        try:
            sql = f"SELECT count(*) FROM t_images_categories_map WHERE category_id={category}"
            self.cursor.execute(sql)
            class_number = self.cursor.fetchall()[0][0]
            incre =  1.0 if number / class_number > 1.0 else number / class_number
            incre = incre / 10
            sql = f"UPDATE t_images_categories_map SET {model_name}_scores = {model_name}_scores + {incre} WHERE category_id={category}"
            self.cursor.execute(sql)
            self.connection.commit()
        except Exception as e:
            print("Batch processing error")
            self.connection.rollback()

    def insert_coco_directory(self, image_dir, coco_json_path, max_prob=1.0):
        """
        批量插入指定目录下所有图片及其COCO格式标注
        """
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        images = coco_data["images"]
        annotations = coco_data["annotations"]
        categories = coco_data["categories"]

        # 创建 id -> 类别名 和 类别名 -> id 的映射
        coco_id2name = {cat["id"]: cat["name"] for cat in categories}

        # 建立图片ID对应的所有标注
        from collections import defaultdict
        image_annotations = defaultdict(list)
        for anno in annotations:
            image_annotations[anno["image_id"]].append(anno)

        try:
            self.connection.begin()

            # 获取数据库已有的类别
            self.cursor.execute("SELECT `name` FROM t_categories")
            existing_categories = {x[0] for x in self.cursor.fetchall()}

            # 获取当前类别id最大值
            self.cursor.execute("SELECT max(`id`) FROM t_categories")
            try:
                category_id = self.cursor.fetchall()[0][0] + 1
            except Exception:
                category_id = 1

            # 插入新类别
            coco_name2id = {}
            for cat in categories:
                name = cat["name"]
                if name not in existing_categories:
                    self.cursor.execute("INSERT INTO t_categories(id, name) VALUES(%s, %s)", (category_id, name))
                    coco_name2id[name] = category_id
                    existing_categories.add(name)
                    category_id += 1
                else:
                    # 先获取旧的id
                    self.cursor.execute("SELECT id FROM t_categories WHERE name=%s", (name,))
                    coco_name2id[name] = self.cursor.fetchone()[0]

            # 获取所有类别映射
            self.cursor.execute("SELECT * FROM t_categories")
            category_name_to_dbid = {row[1]: int(row[0]) for row in self.cursor.fetchall()}

            # 获取当前图像ID最大值
            self.cursor.execute("SELECT max(`id`) FROM t_images")
            try:
                image_id_counter = self.cursor.fetchall()[0][0] + 1
            except Exception:
                image_id_counter = 0

            # 获取关联表字段数量
            self.cursor.execute("DESCRIBE t_images_categories_map")
            num_columns = len(self.cursor.fetchall())

            for image_info in images:
                file_name = image_info["file_name"]
                coco_image_id = image_info["id"]
                file_path = os.path.join(image_dir, file_name)

                if not os.path.exists(file_path):
                    print(f"找不到图像文件: {file_path}")
                    continue

                try:
                    # 加载图像
                    image = Image.open(file_path)

                    # 获取图像的标注
                    annos = image_annotations[coco_image_id]
                    label_names = [coco_id2name[anno["category_id"]] for anno in annos]
                    label_ids = [category_name_to_dbid[name] for name in label_names]

                    # 构造 instance 数据结构
                    instance = {
                        "file_name": file_name,
                        "width": image_info.get("width", image.width),
                        "height": image_info.get("height", image.height),
                        "label": label_ids,
                        "coco_annotations": annos  # 可选，保存原始标注
                    }

                    # 插入图片表
                    sql = "INSERT INTO t_images(id, annotations) VALUES (%s, %s)"
                    self.cursor.execute(sql, (image_id_counter, json.dumps(instance)))

                    # 插入映射表
                    for label_id in set(label_ids):
                        placeholder = (num_columns - 2) * f"{max_prob},"
                        placeholder = placeholder[:-1]
                        sql = f"INSERT INTO t_images_categories_map VALUES (%s, %s, {placeholder})"
                        self.cursor.execute(sql, (image_id_counter, label_id))

                    # 保存图片
                    save_path = IMAGE_ROOT_DIR + str(image_id_counter) + ".jpg"
                    image.save(save_path)

                    image_id_counter += 1

                except Exception as e:
                    print(f"插入图像 {file_name} 时出错：", e)
                    traceback.print_exc()

            self.connection.commit()
            print("所有图像插入完成。")

        except Exception as e:
            print("批量插入失败")
            traceback.print_exc()
            self.connection.rollback()

    def insert_coco_directory(self, image_dir, coco_json_path, max_prob=1.0):
        """
        批量插入指定目录下所有图片及其COCO格式标注
        """
        with open(coco_json_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        images = coco_data["images"]
        annotations = coco_data["annotations"]
        categories = coco_data["categories"]

        # 创建 id -> 类别名 和 类别名 -> id 的映射
        coco_id2name = {cat["id"]: cat["name"] for cat in categories}

        # 建立图片ID对应的所有标注
        from collections import defaultdict
        image_annotations = defaultdict(list)
        for anno in annotations:
            image_annotations[anno["image_id"]].append(anno)

        try:
            self.connection.begin()

            # 获取数据库已有的类别
            self.cursor.execute("SELECT `name` FROM t_categories")
            existing_categories = {x[0] for x in self.cursor.fetchall()}

            # 获取当前类别id最大值
            self.cursor.execute("SELECT max(`id`) FROM t_categories")
            try:
                category_id = self.cursor.fetchall()[0][0] + 1
            except Exception:
                category_id = 1

            # 插入新类别
            coco_name2id = {}
            for cat in categories:
                name = cat["name"]
                if name not in existing_categories:
                    self.cursor.execute("INSERT INTO t_categories(id, name) VALUES(%s, %s)", (category_id, name))
                    coco_name2id[name] = category_id
                    existing_categories.add(name)
                    category_id += 1
                else:
                    # 先获取旧的id
                    self.cursor.execute("SELECT id FROM t_categories WHERE name=%s", (name,))
                    coco_name2id[name] = self.cursor.fetchone()[0]

            # 获取所有类别映射
            self.cursor.execute("SELECT * FROM t_categories")
            category_name_to_dbid = {row[1]: int(row[0]) for row in self.cursor.fetchall()}

            # 获取当前图像ID最大值
            self.cursor.execute("SELECT max(`id`) FROM t_images")
            try:
                image_id_counter = self.cursor.fetchall()[0][0] + 1
            except Exception:
                image_id_counter = 0

            # 获取关联表字段数量
            self.cursor.execute("DESCRIBE t_images_categories_map")
            num_columns = len(self.cursor.fetchall())

            for image_info in images:
                file_name = image_info["file_name"]
                coco_image_id = image_info["id"]
                file_path = os.path.join(image_dir, file_name)

                if not os.path.exists(file_path):
                    print(f"找不到图像文件: {file_path}")
                    continue

                try:
                    # 加载图像
                    image = Image.open(file_path)

                    # 获取图像的标注
                    annos = image_annotations[coco_image_id]
                    label_names = [coco_id2name[anno["category_id"]] for anno in annos]
                    label_ids = [category_name_to_dbid[name] for name in label_names]

                    # 构造 instance 数据结构
                    instance = {
                        "file_name": file_name,
                        "width": image_info.get("width", image.width),
                        "height": image_info.get("height", image.height),
                        "label": label_ids,
                        "coco_annotations": annos  # 可选，保存原始标注
                    }

                    # 插入图片表
                    sql = "INSERT INTO t_images(id, annotations) VALUES (%s, %s)"
                    self.cursor.execute(sql, (image_id_counter, json.dumps(instance)))

                    # 插入映射表
                    for label_id in set(label_ids):
                        placeholder = (num_columns - 2) * f"{max_prob},"
                        placeholder = placeholder[:-1]
                        sql = f"INSERT INTO t_images_categories_map VALUES (%s, %s, {placeholder})"
                        self.cursor.execute(sql, (image_id_counter, label_id))

                    # 保存图片
                    save_path = IMAGE_ROOT_DIR + str(image_id_counter) + ".jpg"
                    image.save(save_path)

                    image_id_counter += 1

                except Exception as e:
                    print(f"插入图像 {file_name} 时出错：", e)
                    traceback.print_exc()

            self.connection.commit()
            print("所有图像插入完成。")

        except Exception as e:
            print("批量插入失败")
            traceback.print_exc()
            self.connection.rollback()

    def insert_instance(self, instance, image, max_prob=1.0):
        """
        插一个样本
        """
        self.connection.begin()
        self.cursor.execute(f"DESCRIBE t_images_categories_map")
        num_columns = len(self.cursor.fetchall())
        try:
            # 检查类别信息, 如果类别出现新类别,那么插入类别表
            categories_table =  "t_categories"
            sql = f"SELECT `name` FROM {categories_table}"
            self.cursor.execute(sql)
            results = [x[0] for x in self.cursor.fetchall()]
            categories_set = set(results)
            sql = f"SELECT max(`id`) FROM {categories_table}"
            self.cursor.execute(sql)
            try:
                category_id = self.cursor.fetchall()[0][0] + 1
            except Exception:
                category_id = 1

            for category in instance["label"]:
                if category in categories_set:
                    continue
                sql = f"INSERT INTO {categories_table}(id, name) VALUES(%s, %s)"
                values = (category_id, category)
                self.cursor.execute(sql, values)
                categories_set.add(category)
                category_id += 1
            # 建立名称和类别的映射
            sql = f"SELECT * FROM {categories_table}"
            self.cursor.execute(sql)
            database_categories_map = { result[1]:int(result[0]) for result in self.cursor.fetchall()}
            # 插入图片数据
            image_table = "t_images"
            image_categories_map_table = "t_images_categories_map"
            sql = f"SELECT max(`id`) FROM {image_table}"
            self.cursor.execute(sql)
            try:
                database_image_id = self.cursor.fetchall()[0][0] + 1
            except Exception:
                database_image_id = 0
            labels = [ x for x in instance["label"]]
            instance["label"] = [database_categories_map[x] for x in labels]
            # 执行第一个插入语句
            sql = f"INSERT INTO {image_table}(id, annotations) VALUES (%s, %s)"
            value = (database_image_id, json.dumps(instance)) 
            self.cursor.execute(sql, value)
            for label in set(labels):
                # 执行第二个插入语句
                placeholder = (num_columns-2) * f"{max_prob},"
                placeholder = placeholder[:-1]
                sql = f"INSERT INTO {image_categories_map_table} VALUES (%s, %s, {placeholder})"
                value = (database_image_id, database_categories_map[label])
                self.cursor.execute(sql, value)

            # 保存图片
            image_save_path =     IMAGE_ROOT_DIR  + str(database_image_id) + ".jpg"
            image.save(image_save_path)
            self.connection.commit()
        except Exception as e:
            print("插入样本失败")
            traceback.print_exc()
            self.connection.rollback()

    def delete_instance_by_image_id(self, image_id):
        """
        根据图片的id删除一个样本
        """
        self.connection.begin()
        try:
            # 删除图片标注信息
            sql = f"DELETE FROM t_images WHERE id=%s"
            self.cursor.execute(sql, [image_id])
            # 删除图片概率映射信息
            sql = f"DELETE FROM t_images_categories_map WHERE image_id=%s"
            self.cursor.execute(sql, [image_id])
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            return False

    @staticmethod
    def get_database_instance():
        if MySQLDatabase.db is None:
            MySQLDatabase.db = MySQLDatabase("403403403")
        return MySQLDatabase.db

if __name__ == "__main__":
    # 声明命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--password", nargs="?", help="password")
    parser.add_argument("--load_annos", type=str,  default=None, help="dataset annotations path ")
    parser.add_argument("--load_images", type=str,  default=None, help="images directory")
    parser.add_argument("--train", action="store_true", help="images directory")


    args = parser.parse_args()
    
    if args.password is None:
        password = getpass.getpass(prompt="Enter password: ")
    else:
        password = args.password
    
    db = MySQLDatabase(password)

    if args.load_annos is not None:
        db.load_annotations(args.load_annos, args.load_images)