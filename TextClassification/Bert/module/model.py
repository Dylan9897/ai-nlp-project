from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Model():
    def __init__(self,args):
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
