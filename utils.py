import argparse


DATA_DIR = "./data"
DATASET_DIR = "./clean_data"

SAVE_DIR = "./saves"




def parse_args():

    description = \
    '''
        args is defined bellow:
    '''         
    parser = argparse.ArgumentParser(description=description)       
    
    parser.add_argument('--learning_rate',help = "learning rate", required=False,type=float,default=1e-5)    
    
    parser.add_argument('--max_epoch',help = "max training epochs", required=False,type=int,default=200)   
    parser.add_argument('--pretrained',help = "use pretrained sememe embeddings", required=False,type=int,default=1)       
    
    parser.add_argument('--mask',help = "use candidate mask", required=False,type=int,default=1) 
    
    parser.add_argument('--tree_attention',help = "use tree attention method", required=False,type=int,default=1)    
    
    parser.add_argument('--depth_method',help = "", required=False,type=str,default="depth")   

    parser.add_argument('--bias_method',help = "", required=False,type=str,default="distance")   
    
    parser.add_argument('--sequence',help = "use sequence encoding result", required=False,type=bool,default=True)    
    
    parser.add_argument('--sememe_data_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/sememes.torch")  
    parser.add_argument('--train_set_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/train.torch")      
    parser.add_argument('--test_set_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/test.torch")      
    parser.add_argument('--valid_set_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/test.torch")     
    parser.add_argument('--model_save_path',help = "", required=False,type=str,default="./model")       
    
    parser.add_argument('--model_name',help = "how to name your model", required=False,type=str)                    
    args = parser.parse_args()                      
    return args