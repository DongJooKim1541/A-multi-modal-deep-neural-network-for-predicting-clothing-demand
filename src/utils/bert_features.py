import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from transformers import BertTokenizer, BertModel


def load_bert(model_name: str = "bert-base-multilingual-cased",
              device: str = "cpu") -> Tuple[BertTokenizer, BertModel]:
    """Load BERT tokenizer and model, move to device."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    net_bert = BertModel.from_pretrained(model_name)
    net_bert = net_bert.to(device)
    net_bert.eval()
    return tokenizer, net_bert


def get_bert_feature(clothing: str, tokenizer: BertTokenizer,
                     net_bert: BertModel, device: str = "cpu") -> torch.Tensor:
    """Extract BERT [CLS] token feature for a single clothing name."""
    encoded_input = tokenizer(clothing, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    with torch.no_grad():
        output = net_bert(**encoded_input)
    feature = output[1].clone().detach()
    return feature


def get_bert_feature_by_batch(clothing_feature: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate pre-extracted BERT features from a batch."""
    out_concat = clothing_feature[0]
    for i in range(1, len(clothing_feature)):
        out_concat = torch.cat((out_concat, clothing_feature[i]), dim=0)
    return out_concat


def build_csv_info(csv_path: Union[str, Path], tokenizer: BertTokenizer,
                   net_bert: BertModel, device: str = "cpu") -> List[Dict[int, torch.Tensor]]:
    """Build dictionary of BERT features indexed by goods_num from CSV."""
    df = pd.read_csv(csv_path, encoding='cp949')
    csv_info = []

    print(f"Extracting BERT features from {len(df)} products...")
    for i in range(len(df)):
        row = df.loc[i]
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
