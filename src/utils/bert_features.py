import torch
import pandas as pd
from transformers import BertTokenizer, BertModel


def load_bert(model_name="bert-base-multilingual-cased", device="cpu"):
    """Load BERT tokenizer and model, move to device."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    net_bert = BertModel.from_pretrained(model_name)
    net_bert = net_bert.to(device)
    net_bert.eval()
    return tokenizer, net_bert


def get_bert_feature(clothing, tokenizer, net_bert, device="cpu"):
    """Extract BERT [CLS] token feature for a single clothing name."""
    encoded_input = tokenizer(clothing, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = net_bert(**encoded_input)
    # [CLS] token pooled output: shape (1, 768)
    feature = output[1].clone().detach()
    return feature


def get_bert_feature_by_batch(clothing_feature):
    """Concatenate pre-extracted BERT features from a batch."""
    for i in range(0, len(clothing_feature)):
        if i == 0:
            out_concat = clothing_feature[0]
        else:
            out_concat = torch.cat((out_concat, clothing_feature[i]), dim=0)
    return out_concat


def build_csv_info(csv_path, tokenizer, net_bert, device="cpu"):
    """Build dictionary of BERT features indexed by goods_num from CSV."""
    df = pd.read_csv(csv_path, encoding='cp949')
    csv_info = []

    print(f"Extracting BERT features from {len(df)} products...")
    for i in range(len(df)):
        row = df.loc[i]
        # CSV columns: index, 1, goodsNum, 1, Name, ...
        goods_num = int(row['index'])
        clothing_name = str(row['Name']) if 'Name' in df.columns else ""

        if clothing_name and clothing_name != 'nan':
            feature = get_bert_feature(clothing_name, tokenizer, net_bert, device)
        else:
            feature = torch.zeros(1, 768)

        csv_info_dict = {goods_num: feature}
        csv_info.append(csv_info_dict)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(df)} features extracted")

    print(f"Completed BERT feature extraction for {len(csv_info)} products")
    return csv_info
