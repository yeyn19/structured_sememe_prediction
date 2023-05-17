import os
from utils import DATA_DIR,DATASET_DIR
import re
from tqdm import tqdm
import torch


class sememe:
    def __init__(self,sememe_name):
        if "|" not in sememe_name:
            assert len(sememe_name) == 1
            self.english, self.chinese = sememe_name, sememe_name
        else:
            self.english, self.chinese = sememe_name.split("|")
        self.description = []
        self.bn = ""
        self.wn = ""
        self.vector = None


    def __repr__(self) -> str:
        return f"[{self.english}|{self.chinese}]"
    
    def __eq__(self, value: object) -> bool:
        if type(value) != sememe:
            return False
        return (self.english == value.english) and (self.chinese == value.chinese)


class my_sememe_tree:
    def __init__(self):
        self.childs = []
        self.line_type = ""
        self.father = None #根节点father是None
        # pass
        self.info = {"sememe":sememe("BACK|BACK")}



    def print(self,start_space=0):
        '''
        打印
        '''
        print(" "*start_space, end="")
        prefix = f"|--{self.line_type}-->"
        print(f"{prefix}{self.info['sememe']}",end="")
        print("")
        for child in self.childs:
            child.print(start_space = start_space + len(prefix))


    def to_tree_sequence(self,add_eos = False):
        tree_sequence = []

        tree_sequence.append(self.info["sememe"])
        for child in self.childs:
            tree_sequence.extend(child.to_tree_sequence(add_eos = False))
            tree_sequence.append(sememe("BACK|BACK"))
        if add_eos:
            tree_sequence.append(sememe("EOS|EOS"))
        return tree_sequence
    

def pseudo_tokenizer(sememes,):
    pass

def parse_sememe(global_data,sememes, line: str):
    # print(line)
    wn, bn, sememe_tree_line = line.split("	")
    # print(wn)
    # print(bn)
    # print(sememe_tree_line)

    sememe_tree_lines = sememe_tree_line[1:].split(";")
    for k, text in enumerate(sememe_tree_lines):
        text_list = [char for char in text]
        next_token = '{'
        for id in range(len(text_list)):
            if text_list[id] == "\"":
                text_list[id] = next_token
                next_token = "}" if next_token == "{" else "{"
        text = "".join(text_list)

        # print(text)
        root = recursive_deal_sememe_line(text) #最前面是"="符号

        # root[0].print(start_space=0)

        assert len(root) == 1
        root = root[0]

        for instance in root.to_tree_sequence():
            # if "{" in instance.__repr__() or "}" in instance.__repr__():
            #     assert "}{" in instance.__repr__()
            #     print(text)
            #     root.print()
            #     exit()
            if sememes.get(instance.__repr__(),-1) == -1:
                sememes[instance.__repr__()] = instance   

        if global_data.get(bn,-1) == -1:
            global_data[bn] = []
        global_data[bn].append(root)

        sememes[root.to_tree_sequence()[0].__repr__()].bn = bn
        sememes[root.to_tree_sequence()[0].__repr__()].wn = wn

    # exit()

def recursive_deal_sememe_line(line): #return my_sememe_tree
    assert line[0] == "{" and line[-1] == "}"
    line = line[1:-1]
    '''
    使用括号匹配算法，先找到每一组匹配的括号
    '''
    pattern = r"([^:]+)(:.+)?"
    re_result = re.match(pattern,line)
    assert re_result != None
    # print(re_result.groups())
    sememe_name = re_result.groups()[0]
    sememe_names = sememe_name.split("}{")
    nodes = []
    for name in sememe_names:
        node = my_sememe_tree()
        node.info["sememe"] = sememe(name)
        if re_result.groups()[1] != None: #有子义原
            # print(re_result.group(2)) #
            string = re_result.group(2)[1:]
            data_blocks = []
            pos = 0
            block_start_with = 0
            bracket = 0
            while pos < len(string):
                if string[pos] == "{":
                    bracket += 1
                elif string[pos] == "}":
                    bracket -= 1
                elif string[pos] == ",":
                    if bracket == 0:
                        data_blocks.append(string[block_start_with:pos])
                        block_start_with = pos + 1
                pos += 1
            data_blocks.append(string[block_start_with:pos])
            for data in data_blocks:
                child_sememe_pattern = r"^([^\{\:\}]+?=)?(.*)$" #(,([^\{:\}]+?=)?(\{.+?\}))*
                child_match_result = re.match(child_sememe_pattern,data)
                assert child_match_result != None
                childs = recursive_deal_sememe_line(child_match_result.groups()[1])
                for child in childs:
                    child.line_type = child_match_result.groups()[0] if child_match_result.groups()[0] != None else ""
                    if child.line_type != "":
                        assert child.line_type[-1] == "="
                        child.line_type = child.line_type[:-1]
                    node.childs.append(child)
        nodes.append(node)
    return nodes


            # print(child_match_result.groups())


def get_sememe_trees():
    input_dir = os.path.join(DATA_DIR,"synsetStructed.txt")
    good = 0
    total = 0
    global_data = {}
    sememes = {}
    with open(input_dir,"r",encoding="utf-8") as f:
        for line in tqdm(f.readlines(),total=len(f.readlines())):
            total += 1
            # parse_sememe(global_data, sememes, line.strip())
            try:
                parse_sememe(global_data,sememes, line.strip())
                good += 1
            except:
                # print(line.strip())
                pass
    print(f"ratio={good/total}, good={good}, total={total}")


    pad_sememe = sememe("PAD|PAD")
    sememes[pad_sememe.__repr__()] = pad_sememe

    eos_sememe = sememe("EOS|EOS")
    sememes[eos_sememe.__repr__()] = eos_sememe

    print(f"find {len(sememes)} sememes")


    return global_data, sememes

def get_description_maps(sememes):
    with open(os.path.join(DATA_DIR,"synsetDef.txt"),"r",encoding="utf-8") as f:
        data = f.readlines()
        # print(len(data))
        assert len(data) % 4 == 0
        total_description = 0
        find_sememes = 0
        for i in tqdm(range(len(data) // 4)):

            bn = data[4*i].strip()
            description = data[4*i+1].strip().split("	")[1:]
            if bn != "":
                total_description += 1
                find  = False
                for key, value in sememes.items():
                    if value.bn == bn:
                        sememes[key].description = description
                        find = True
                        break
                if not find:
                    pass
                else:
                    find_sememes += 1
    print(f"find {total_description} descriptions, {find_sememes} are in sememe datasets")

    return sememes

def load_vector(sememes):
    sememe_vectors = {}
    with open(os.path.join(DATA_DIR,"sememe-vec.txt"),"r",encoding="utf-8") as f:
        data = f.readlines()[1:]
        for instance in tqdm(data):
            split = instance.split(" ")
            sememe_name = split[0]
            sememe_vector = split[1:-1]
            sememe_vector = [float(x) for x in sememe_vector]
            sememe_vector = torch.tensor(sememe_vector)
            sememe_vectors[sememe_name] = sememe_vector
        total_sememe = len(sememe_vectors)
        find_sememe = 0
        for key in sememes.keys():
            if sememes[key].chinese in sememe_vectors.keys() != -1:
                find_sememe += 1
                sememes[key].vector = sememe_vectors[sememes[key].chinese]
        print(f"{total_sememe} sememes have vector, {find_sememe} are in dataset")
    return sememes


def preprocess_data():
    global_data, sememes = get_sememe_trees()
    sememes = get_description_maps(sememes)
    sememes = load_vector(sememes)

    keys = list(global_data.keys())
    train_keys = keys[:int(0.95*len(keys))]

    trainset = {key: global_data[key] for key in train_keys}
    testset = {key: global_data[key] for key in global_data.keys() if key not in train_keys}

    torch.save(sememes,os.path.join(DATASET_DIR,"sememes.torch"))
    torch.save(trainset,os.path.join(DATASET_DIR,"trainset.torch"))
    torch.save(testset,os.path.join(DATASET_DIR,"testset.torch"))
    return global_data, sememes

if __name__ == "__main__":
    preprocess_data()