import os,time,sys,cv2
import google.generativeai as genai
from collections import defaultdict
import json
import numpy as np
import PIL.Image
genai.configure(api_key="")

# python /Users/abemasatoshi/Downloads/kidneyGemini/gemini_demo_MIL_oof_KNN.py
DIR = "/Users/abemasatoshi/Downloads/kidneyGemini/kid/kidney"

wo_table = 0
re_think = 0
revserse_order = 0
Hflip_image = 0
Vflip_image = 0
add_other_agent = 0
CoT = 1
K = 1
modelname = "uni" #"resnet50" "vit" "dinoglo"
seed =int(sys.argv[2])
if not CoT:
    EXP_NAME = f"KNN_{modelname}"+"wotable:"+str(wo_table)+"rethink:"+str(re_think)+"reverse_order:"+str(revserse_order)+"Hflip:"+str(Hflip_image)+"Vflip:"+str(Vflip_image)+"add_other_agent:"+str(add_other_agent)+"CoT:"+str(CoT)+"few:"+str(K)+"seed:"+str(seed)
else:
    EXP_NAME = f"KNN_{modelname}"+"wotable:"+str(wo_table)+"rethink:"+str(re_think)+"reverse_order:"+str(revserse_order)+"Hflip:"+str(Hflip_image)+"Vflip:"+str(Vflip_image)+"add_other_agent:"+str(add_other_agent)+"few:"+str(K)+"seed:"+str(seed)
SAVE_DIR = f"/Users/abemasatoshi/Downloads/kidneyGemini/{EXP_NAME}"

## save this file to SAVE_DIR
import os,shutil
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR,exist_ok=True)
if not os.path.exists(f"{SAVE_DIR}/exp.py"):
    shutil.copy("/Users/abemasatoshi/Downloads/kidneyGemini/gemini__KNN_mean.py",f"{SAVE_DIR}/exp.py")
    
import pandas as pd
df = pd.read_csv(f"{DIR}/sub_exp__4cls_dino.csv")
print(df.columns)
print(df.head())

folds = pd.read_csv(f"{DIR}/oof_exp__4cls_dino.csv")

df = pd.concat([df,folds],axis=0).reset_index(drop=True)
missing = pd.read_csv(f"/Users/abemasatoshi/Downloads/kidneyGemini/kidney_missing.csv")
def get_missing(df):
    cols = ["BMI","DBP","DM","age","alb","egfr","uprot","収縮期血圧","血尿"]
    df_tmp = df[df["WSI"].isin(missing["label"].values)]#.reset_index(drop=True)
    df_full = df[df["WSI"].isin(missing["label"].values)==False]#.reset_index(drop=True)
    #male,age,SBP,DBP,BMI,egfr,alb,upcr,OB_teisei,dm,DM1,DM
    bmi = dict(zip(missing["label"],missing["BMI"])) 
    df_tmp["BMI"] = df_tmp["WSI"].map(bmi)
    dbp = dict(zip(missing["label"],missing["DBP"]))
    df_tmp["DBP"] = df_tmp["WSI"].map(dbp)
    dm = dict(zip(missing["label"],missing["DM"]))
    df_tmp["DM"] = df_tmp["WSI"].map(dm)
    age = dict(zip(missing["label"],missing["age"]))
    df_tmp["age"] = df_tmp["WSI"].map(age)
    alb = dict(zip(missing["label"],missing["alb"]))
    df_tmp["alb"] = df_tmp["WSI"].map(alb)
    egfr = dict(zip(missing["label"],missing["egfr"]))
    df_tmp["egfr"] = df_tmp["WSI"].map(egfr)
    uprot = dict(zip(missing["label"],missing["upcr"]))
    df_tmp["uprot"] = df_tmp["WSI"].map(uprot)
    sbp = dict(zip(missing["label"],missing["SBP"]))
    df_tmp["収縮期血圧"] = df_tmp["WSI"].map(sbp)
    ketu = dict(zip(missing["label"],missing["OB_teisei"]))
    df_tmp["血尿"] = df_tmp["WSI"].map(ketu)
    df = pd.concat([df_tmp,df_full],axis=0)
    return df



df = get_missing(df)

if modelname == "resnet50":
    df_emb = np.load("/resnet50_embeddings.npy")
elif modelname == "vit":
    df_emb = np.load("./vit_embeddings.npy")
elif modelname == "dinoglo":
    df_emb = np.load("./dinoglo_embeddings.npy")
elif modelname == "uni":
    df_emb = np.load("./uniV1_embeddings.npy")

import faiss
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # ベクトルの次元数
    index = faiss.IndexFlatL2(dimension)  # L2ノルム（ユークリッド距離）で検索
    index.add(embeddings)  # 埋め込みをインデックスに追加
    return index

# 2. k近傍検索
def faiss_knn_search(index, query_embedding, k=5):
    distances, indices = index.search(query_embedding.reshape(1, -1), k)  # クエリを検索
    return indices[0], distances[0]


df_index = build_faiss_index(df_emb)

json_path =f"{SAVE_DIR}/suboof.json"
json_path_raw = f"{SAVE_DIR}/suboof_raw.json"
json_path_analyze = f"{SAVE_DIR}/suboof_analyze.json"
existing_results = defaultdict(str) 
existing_results_raw = defaultdict(str)
existing_results_analyze = defaultdict(str)
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        existing_results.update(json.load(f))
if os.path.exists(json_path_raw):
    with open(json_path_raw, 'r', encoding='utf-8') as f:
        existing_results_raw.update(json.load(f))
if os.path.exists(json_path_analyze):
    with open(json_path_analyze, 'r', encoding='utf-8') as f:
        existing_results_analyze.update(json.load(f))



#sample_file_1 = PIL.Image.open(image_path_1)
txt = []
model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
# each wsi id
disease_labels= {
    "MCN": "Minimal change nephrotic syndrome",
    "MSP": "Mesangial proliferative glomerulonephritis (include IgA nephropathy)",
    "MEN": "Membranous nephropathy",
    "DMN": "Diabetic nephropathy",
    }
label2labelname = {0:"MCN",1:"MEP",2:"MEN",3:"DMN"}
def get_few(df,few,seed):
    new_ =[]
    few_each_ = []
    for col in [0,1,2,3]:
        tmp_ = df[df["label"]==col].sample(n=few,random_state=seed)
        new_.append(tmp_)
        few_each_.append(len(tmp_))
    return pd.concat(new_,axis=0).reset_index(drop=True),few_each_
for IDX in df["WSI"].unique()[:]:
    if IDX in existing_results:
        continue

    df_tmp = df[df["WSI"]==IDX]#.reset_index(drop=True)
    tmp = df_tmp.iloc[0]
    test_index = df_tmp.index
    test_emb_ = df_emb[test_index]
    image_paths= []
    for i in df_tmp["path"].values:
        i = i.split("/")[-1]    
        image_paths.append(f"{DIR}/{i}")
    
    
    DBP = tmp["DBP"]
    egfr = tmp["egfr"]
    DM = tmp["DM"]
    upc = tmp["uprot"]
    ketu = tmp["血尿"]
    age = tmp["age"]
    alb = tmp["alb"]
    BMI = tmp["BMI"]
    sbp = tmp["収縮期血圧"]

    sample_file_2 = [PIL.Image.open(i) for i in image_paths]
    fewshot = []
    examplelabel = []

    test_emb_mean = test_emb_.mean(axis=0)
    indices, distances = faiss_knn_search(df_index, test_emb_mean, k=70)
    for ind in indices:
        if ind in test_index:
            continue
        fewshot.append(df.iloc[ind]["path"])
        examplelabel.append(label2labelname[df.iloc[ind]["label"]])

    image_paths_few= []
    fewshot = fewshot[:K]
    examplelabel = examplelabel[:K]
    for i in fewshot:
        i = i.split("/")[-1]    
        image_paths_few.append(f"{DIR}/{i}")
    sample_file_few = [PIL.Image.open(i) for i in image_paths_few]
    if revserse_order:
        sample_file_2 = sample_file_2[::-1]
    if Hflip_image:
        """# error
        google.api_core.exceptions.InvalidArgument: 400 Request payload size exceeds the limit: 20971520 bytes. The file size is too large. Please use the File API to upload your files instead. Example: `f = genai.upload_file(path); m.generate_content(['tell me about this file:', f])`
        """
        sample_file_2 = [i.transpose(PIL.Image.FLIP_LEFT_RIGHT) for i in sample_file_2]
    if Vflip_image:
        sample_file_2 = [i.transpose(PIL.Image.FLIP_TOP_BOTTOM) for i in sample_file_2]
    if K>0:
        sample_file_2 = sample_file_few+sample_file_2
    #Choose a Gemini model.
    time.sleep(1)
    

    prompt = f"There are PAS stain images of kidney glomerulus. think about image findings and possible diseases ,disease_labels are {disease_labels},"
    if not wo_table:
        prompt+=f"clinical infomation:DBP={DBP},sBP = {sbp}, eGFR={egfr},has DM or not={DM} uro protein={upc},{age} years old,alb={alb},occult hematuria={ketu},BMI = {BMI}"

    if CoT:
        prompt+="think step by step to improve your diagnostic accuracy,first,think about the image findings and possible diseases,then think about the patient's clinical information,finally,think about the patient's clinical information and the image findings together"
    prompt+="and then output which of 4 diseases is most likely to be present in the patient."
    if re_think:
        prompt+="After calculating the predicted value, recheck the clinical information and image information to determine whether the predicted value is correct, and if necessary, revise the prediction."
    if K>0:
        prompt+=f"first {len(fewshot)} images are example,example labels are {examplelabel},the others are what I want you to classify,example images are similar to the images I want you to classify,OUTPUT is only labelname & probarility,do not output other text"

    prompt+="""EXAMPLE:OUTPUT={'MCN':0.XXX,'MSP':0.XXX,'MEN':0.XXX,'DMN':0.XXX},sum of 4 class probarility has to be 1,output is only labelname & probarility,dnt output other text"""
    

    # save prompt to txt
    with open(f"{SAVE_DIR}/prompt.txt", "w") as f:
        f.write(prompt)
    try:
        response = model.generate_content([prompt, *sample_file_2])
    except Exception as e:
        
        response = model.generate_content([prompt, *sample_file_2])
        print(e)
        print(IDX)
        exit()
    out = response.text
    out = out.replace("\n","").replace(" ","")
    
    for cls in ["MCN","MEN","MSP","DMN"]:
        if "{'"+cls in out:
            out = "{'"+cls+out.split("{'"+cls)[-1]
            out = out.split("}")[0]+"}"
   
    txt.append(response.text)
    existing_results[IDX] = out
    existing_results_raw[IDX] = response.text
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dict(existing_results), f, ensure_ascii=False, indent=2)
    with open(json_path_raw, 'w', encoding='utf-8') as f:
        json.dump(dict(existing_results_raw), f, ensure_ascii=False, indent=2)

## calculate ROC_AUC
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def calculate_4class_auc(json_path):
    # Load predictions
    with open(json_path, 'r') as f:
        predictions = json.load(f)
    
    # Initialize lists
    y_true = []
    y_pred = []
    
    # Process each case
    for case_id, pred_str in predictions.items():
        pred_str = pred_str.replace("'", '"')  # Replace single quotes with double quotes
        pred_str = pred_str.replace("{M", '{"M')  # Replace NaN with 0.0
        pred_str = pred_str.split("}")[0] + "}"  # Remove trailing characters
        pred_str = "{"+pred_str.split("{")[-1] # Replace NaN with 0.0
        try:
            pred_dict = json.loads(pred_str)
        except:
            print(pred_str)
        try:
            # Clean up string and parse as JSON
            
            
            pred_dict = json.loads(pred_str)
            # Get predictions
            pred_probs = [
                float(pred_dict['MCN']),
                float(pred_dict['MSP']),
                float(pred_dict['MEN']),
                float(pred_dict['DMN'])
            ]
            
            # Get true label
            if case_id.startswith(('IGA_', 'PRU_')):
                true_label = 'MSP'
            elif case_id.startswith('DMN_'):
                true_label = 'DMN'
            elif case_id.startswith('MEN_'):
                true_label = 'MEN'
            else:
                true_label = 'MCN'
            
            y_pred.append(pred_probs)
            y_true.append(true_label)
            
        except Exception as e:
            print(f"Error processing {case_id}: {str(e)}")
            continue

     
       
    # Convert to binary format
    classes = ['MCN', 'MSP', 'MEN', 'DMN']
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred = np.array(y_pred)
    print(len(y_true_bin),len(y_pred))

    # Calculate AUC for each class
    auc_scores = {}
    for i, cls in enumerate(classes):
        auc = roc_auc_score(y_true_bin[:, i], y_pred[:, i])
        auc_scores[cls] = auc
    auc_scores["mean"] = np.mean(list(auc_scores.values()))

    # Print results
    print("\nROC AUC Scores:")
    ## save socre to json
    with open(f"{SAVE_DIR}/auc_scores.json", 'w', encoding='utf-8') as f:
        json.dump(auc_scores, f, ensure_ascii=False, indent=2)
    print("-" * 30)
    for cls, score in auc_scores.items():
        print(f"{cls:.<20} {score:.4f}")
    print("-" * 30)
    print(f"Macro Average:........ {np.mean(list(auc_scores.values())):.4f}")

    return auc_scores


auc_scores = calculate_4class_auc(json_path)
