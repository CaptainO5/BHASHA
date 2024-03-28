import evaluate

def bleu_score(predictions, references):
    bleu = evaluate.load('bleu')
    return bleu.compute(predictions=predictions, references=references)['bleu']

def rouge_scores(predictions, references):
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=predictions, references=references)

def pinc_score():
    pass
