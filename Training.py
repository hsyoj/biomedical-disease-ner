import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from datasets import DatasetDict, enable_progress_bar, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
    TrainingArguments, Trainer, EarlyStoppingCallback
from torchcrf import CRF
import numpy as np
import evaluate
import os

class TokenClassifierWithCRF(nn.Module):
    def __init__(self, checkpoint, num_labels, id2label, label2id):
        super().__init__()
        self.base = AutoModelForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        self.crf = CRF(num_labels, batch_first=True)
        self.config = self.base.config

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits

        loss = None
        if labels is not None:
            device = logits.device
            mask = labels.ne(-100).to(device)
            tags = labels.masked_fill(~mask, 0).to(device)

            try:
                o_id = self.base.config.label2id['O']
            except Exception:
                o_id = 0

            mask0 = mask.clone()
            tags0 = tags.clone()
            mask0[:, 0] = True
            tags0[:, 0] = o_id

            loss = -self.crf(logits, tags0, mask=mask0, reduction='mean')

        return {"loss": loss, "logits": logits}

class BertTrainer:
    def __init__(self, model_path, data_suffix, data_dir='processed', bs=16, lr=2e-5, eps=40,
                 label_all_tokens = True, lower_case:bool|None = None, crf = False, freeze = 4):
        self.checkpoint = model_path
        self.batch_size = bs
        self.learning_rate = lr
        self.num_epochs = eps
        self.suffix = data_suffix
        self.data_dir = data_dir
        self.data_files = self._get_datafile_dict()
        self.label_all_tokens = label_all_tokens
        self.lower_case = self.suffix in {"Biomed", "Sap"} if lower_case is None else lower_case
        self.n_freeze = freeze
        self.use_crf = crf
        self._load_dataset()
        self._init_labels_from_data()
        self._init_model()

    def _get_datafile_dict(self):
        base = os.path.join(self.data_dir, "")
        return {
            "train":      base + f"NCBI_train_{self.suffix}.conll",
            "validation": base + f"NCBI_dev_{self.suffix}.conll",
            "test":       base + f"NCBI_test_{self.suffix}.conll",
        }

    def _init_labels_from_data(self):
        splits = ["train", "validation", "test"]
        unique_labels = sorted({
            lab for sp in splits for example in self.dataset[sp] for lab in example["ner_tags"]
        })
        self.label_list = unique_labels
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

    def _read_conll(self, path):
        tokens, labels, sentences = [], [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    if tokens:
                        sentences.append({"tokens": tokens, "ner_tags": labels})
                        tokens, labels = [], []
                    continue
                tok, lab = line.split('\t')
                tokens.append(tok)
                labels.append(lab)
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": labels})
        return Dataset.from_list(sentences)

    def _load_dataset(self):
        self.dataset = DatasetDict({
            split: self._read_conll(path) for split, path in self.data_files.items()
        })

    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=512,
            padding=False
        )

        labels = []
        for i in range(len(examples["tokens"])):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id < len(examples["ner_tags"][i]):
                    orig_tag = examples["ner_tags"][i][word_id]
                    if self.label_all_tokens and word_id == prev_word_id and orig_tag.startswith("B-"):
                        orig_tag = "I-" + orig_tag[2:]
                    if word_id != prev_word_id or self.label_all_tokens:
                        label_ids.append(self.label2id[orig_tag])
                    else:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                prev_word_id = word_id
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _freeze_bottom(self, n_layers):
        model = self.model.base if hasattr(self.model, "base") else self.model
        if hasattr(model, "bert"):
            for p in model.bert.embeddings.parameters():
                p.requires_grad = False
            for i in range(n_layers):
                for p in model.bert.encoder.layer[i].parameters():
                    p.requires_grad = False

    def _init_model(self):
        num_labels  = len(self.label_list)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, do_lower_case=self.lower_case)
        if self.use_crf:
            self.model = TokenClassifierWithCRF(
                self.checkpoint,
                num_labels=num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.checkpoint,
                num_labels=num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self._freeze_bottom(n_layers=self.n_freeze)

    def _compute_metrics(self, eval_pred):
        metric = evaluate.load("seqeval")
        logits, labels = eval_pred
        mask = labels != -100

        if hasattr(self.model, "crf"):
            with torch.no_grad():
                model_device = next(self.model.parameters()).device
                emissions = torch.tensor(logits, dtype=torch.float32, device=model_device)
                mask_t    = torch.tensor(mask,  dtype=torch.bool,     device=model_device)

                mask0 = mask_t.clone()
                mask0[:, 0] = True

                paths = self.model.crf.decode(emissions, mask=mask0)

            preds = np.full(labels.shape, fill_value=-100, dtype=np.int64)
            for i, path in enumerate(paths):
                path = path[1:]
                valid_cols = np.where(mask[i])[0]
                take = min(len(path), len(valid_cols))
                preds[i, valid_cols[:take]] = np.array(path[:take], dtype=np.int64)
        else:
            preds = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        total, correct = 0, 0
        for lab_seq, pred_seq in zip(labels, preds):
            tl, tp = [], []
            for l, p in zip(lab_seq, pred_seq):
                if l != -100:
                    tl.append(self.id2label[int(l)])
                    tp.append(self.id2label[int(p)])
                    total += 1
                    if l == p:
                        correct += 1
            if tl:
                true_labels.append(tl)
                true_preds.append(tp)

        res = metric.compute(predictions=true_preds, references=true_labels, zero_division=0)
        return {
            "precision": res["overall_precision"],
            "recall":    res["overall_recall"],
            "f1":        res["overall_f1"],
            "accuracy":  correct / total if total else 0.0,
        }

    def _plot_metrics(self, log_history):
        train_losses = [x["loss"]      for x in log_history if "loss"      in x]
        eval_f1s     = [x["eval_f1"]   for x in log_history if "eval_f1"   in x]
        eval_accs    = [x["eval_accuracy"] for x in log_history if "eval_accuracy" in x]

        plt.figure()
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.show()

        plt.figure()
        plt.plot(eval_f1s)
        plt.title("Eval F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.show()

        plt.figure()
        plt.plot(eval_accs)
        plt.title("Eval Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.show()

    def _print_results(self, train_res, test_metrics):
        print("\n===== Training Results =====")
        for key, value in train_res.metrics.items():
            print(f"{key}: {value}")
        print("\n===== Test Metrics =====")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")

    def train_eval(self):
        enable_progress_bar()

        tokenized = self.dataset.map(
            self._tokenize_and_align_labels,
            batched=True,
            remove_columns=["tokens", "ner_tags"],
            desc=f"Tokenizing ({self.suffix})"
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        args = TrainingArguments(
            output_dir='results',
            eval_strategy='epoch',
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            logging_steps=50,
            disable_tqdm=False,
            report_to="none",
            max_grad_norm=1.0,
            save_total_limit=2,
            seed=42,
            group_by_length=True,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            lr_scheduler_type='linear',
            #fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        train_res = trainer.train()
        test_metrics = trainer.evaluate(tokenized["test"])
        trainer.save_model(f"saved_models/{self.suffix}")
        self._print_results(train_res, test_metrics)
        self._plot_metrics(trainer.state.log_history)

def get_bert_models():
    local_models = {
        "Biomed"   : "./BiomedNLP_BiomedBERT",
        "Bio"      : "./BioBERT",
        "Sap"      : "./SapBERT",
        "Clinical" : "./Bio_ClinicalBERT"
    }
    online_models = {
        "Biomed"   : "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
        "Bio"      : "dmis-lab/biobert-base-cased-v1.2",
        "Sap"      : "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "Clinical" : "emilyalsentzer/Bio_ClinicalBERT"
    }
    return {
        i: (local_models[i] if os.path.exists(local_models[i]) else online_models[i])
        for i in online_models
    }

if __name__ == "__main__":
    bert_models = get_bert_models()

    for suffix, cp_path in bert_models.items():
        print("="*60)
        print('-'*20, f"{suffix}BERT", '-'*20)
        bt = BertTrainer(model_path=cp_path, data_suffix=suffix, label_all_tokens=False, lr=3e-5, freeze=0, crf=False)
        bt.train_eval()
        print("="*60, '\n\n')
