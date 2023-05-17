import argparse



DATA_DIR = "./data"
DATASET_DIR = "./clean_data"

SAVE_DIR = "./saves"

LOG_DIR = "./tensorboard"





def parse_args():

    description = \
    '''
        args is defined bellow:
    '''         
    parser = argparse.ArgumentParser(description=description)       
    

    parser.add_argument('--exp_name',help = "", required=False,type=str,default="TSTG") 
    parser.add_argument('--learning_rate',help = "learning rate", required=False,type=float,default=1e-5)   
    parser.add_argument('--weight_decay',help = "weight_decay", required=False,type=float,default=1e-2)   
    parser.add_argument('--betas',help = "beta", required=False,type=float,default=0.95)    
    parser.add_argument('--train_batch_size',help = "train batch_size", required=False,type=int,default=8)    
    parser.add_argument('--eval_batch_size',help = "eval batch_size", required=False,type=int,default=16)    
    parser.add_argument('--max_epoch',help = "max training epochs", required=False,type=int,default=50)   

    parser.add_argument('--save_steps',help = "max training epochs", required=False,type=int,default=3000)   
    parser.add_argument('--eval_steps',help = "max training epochs", required=False,type=int,default=100)   
    # parser.add_argument('--log_steps',help = "max training epochs", required=False,type=int,default=5)   

    parser.add_argument('--grad_clip',help = "grad_clip", required=False,type=float,default=1.0) 

    parser.add_argument('--ckpt_path',help = "", required=False,type=str,default=fr"C:\git_repos\structured_sememe_prediction\saves\TSTG\3000.pt")
    parser.add_argument('--test_only',help = "", required=False,type=bool,default=f"False")

    parser.add_argument('--pretrained',help = "use pretrained sememe embeddings", required=False,type=int,default=1)       
    
    parser.add_argument('--mask',help = "use candidate mask", required=False,type=int,default=1) 
    
    parser.add_argument('--tree_attention',help = "use tree attention method", required=False,type=int,default=1)    
    
    parser.add_argument('--depth_method',help = "", required=False,type=str,default="depth")   

    parser.add_argument('--bias_method',help = "", required=False,type=str,default="distance")   
    
    parser.add_argument('--sequence',help = "use sequence encoding result", required=False,type=bool,default=True)    
    
    parser.add_argument('--sememe_data_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/sememes.torch")  
    parser.add_argument('--train_set_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/trainset.torch")      
    parser.add_argument('--test_set_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/testset.torch")      
    parser.add_argument('--valid_set_path',help = "", required=False,type=str,default=f"{DATASET_DIR}/testset.torch")     
    parser.add_argument('--model_save_path',help = "", required=False,type=str,default="./model")       
    
    parser.add_argument('--model_name',help = "how to name your model", required=False,type=str)                    
    args = parser.parse_args()                      
    return args