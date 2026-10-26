import transformers as tf
import os

def read_ncbi_file(filename):
    docs = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        if '|t|' in lines[i]:
            pmid = lines[i].split('|')[0]
            title = lines[i].strip().split('|t|')[1]
            abstract = lines[i+1].strip().split('|a|')[1] if '|a|' in lines[i+1] else ""
            text = title + " " + abstract
            entities = []
            j = i+2
            while j < len(lines) and lines[j].strip() and '|t|' not in lines[j] and '|a|' not in lines[j]:
                parts = lines[j].strip().split('\t')
                if len(parts) >= 5:
                    ent_start = int(parts[1])
                    ent_end = int(parts[2])
                    ent_type = parts[4].strip().split()[0]
                    entities.append((ent_start, ent_end, ent_type))
                j += 1
            docs.append({'pmid': pmid, 'text': text, 'entities': entities})
            i = j
        else:
            i += 1
    return docs

def make_char_bio(text, entities):
    bio_labels = ['O'] * len(text)
    for start, end, ent_type in entities:
        if end > len(text):
            end = len(text)
        for idx in range(start, end):
            if idx == start:
                bio_labels[idx] = 'B-' + ent_type
            else:
                bio_labels[idx] = 'I-' + ent_type
    return bio_labels

def convert_to_conll(docs, tokenizer, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in docs:
            text = doc['text']
            char_bio = make_char_bio(text, doc['entities'])
            encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding['offset_mapping']
            input_ids = encoding['input_ids']

            for idx, (start, end) in enumerate(offsets):
                if end > start:
                    token = tokenizer.convert_ids_to_tokens([input_ids[idx]])[0]
                    token_labels = char_bio[start:end]
                    if all(l == 'O' for l in token_labels):
                        label = 'O'
                    else:
                        label = [l for l in token_labels if l != 'O'][0]
                    f.write(f"{token}\t{label}\n")
            f.write("\n")

def convert_ncbi_corpus(model_name: str, suffix: str, do_lower_case: bool):
    print(f"\n>>> Using model: {model_name}")
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name, do_lower_case=do_lower_case)

    for src, split in [
        ("NCBItrainset_corpus.txt", "train"),
        ("NCBIdevelopset_corpus.txt", "dev"),
        ("NCBItestset_corpus.txt", "test"),
    ]:
        output_file = os.path.join(".", "processed", f"NCBI_{split}_{suffix}.conll")
        src_path = os.path.join(".", "data", src)
        try:
            docs = read_ncbi_file(src_path)
            convert_to_conll(docs, tokenizer, output_file)
            print(f"{src_path} → {output_file} conversion is successful！")
        except Exception as e:
            print(f"{src_path} conversion failed: {e}")

def get_bert_models():
    local_models = {
        "Biomed"   : "./BiomedNLP_BiomedBERT",
        "Sap"      : "./SapBERT",
        "Bio"      : "./BioBERT",
        "Clinical" : "./Bio_ClinicalBERT"
    }
    online_models = {
        "Biomed"   : "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        "Sap"      : "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "Bio"      : "dmis-lab/biobert-base-cased-v1.2",
        "Clinical" : "emilyalsentzer/Bio_ClinicalBERT"
    }
    return {
        i: (local_models[i] if os.path.exists(local_models[i]) else online_models[i])
        for i in online_models
    }

if __name__ == "__main__":
    bert_models = get_bert_models()
    for suffix, model_path in bert_models.items():
        convert_ncbi_corpus(model_path, suffix, suffix in {"Biomed", "Sap"})
