from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import logging
import sys
import argparse
import sagemaker
import os
import torch
import os, requests, json, boto3
from transformers import DistilBertTokenizer
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering, Trainer, TrainingArguments


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def get_data():
    if not os.path.exists('squad'):
        os.mkdir('squad')

    url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
    res = requests.get(f'{url}train-v2.0.json')

    # loop through
    for file in ['train-v2.0.json', 'dev-v2.0.json']:
        # make the request to download data over HTTP
        res = requests.get(f'{url}{file}')
        # write to file
        with open(f'./data/{file}', 'wb') as f:
            for chunk in res.iter_content(chunk_size=4):
                f.write(chunk)        

def read_squad(path):
    # open JSON file and load intro dictionary
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                # check if we need to be extracting from 'answers' or 'plausible_answers'
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa[access]:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers

def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
                    
# convert our character start/end positions to token start/end positions
def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})                    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str,default='distilbert-base-uncased-distilled-squad')
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    
    # Data, model, and output directories;
    parser.add_argument("--output_data_dir", type=str, default='output_data')
    parser.add_argument("--model_dir", type=str, default='model_dir')
    parser.add_argument("--n_gpus", type=str, default=1)
    parser.add_argument("--training_dir", type=str, default='training')
    parser.add_argument("--test_dir", type=str, default='test_dir')
    

    args, _ = parser.parse_known_args()

    model_name=args.model_name
    
    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    get_data()

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
      
    train_file = "./data/train-v2.0.json"
    test_file = "./data/dev-v2.0.json"
       
    # data processing
    # execute our read SQuAD function for training and validation sets
    train_contexts, train_questions, train_answers = read_squad(train_file)
    val_contexts, val_questions, val_answers = read_squad(test_file)
    
    #get the character position at which the answer ends in the passage
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)
    
    # Use only a subset of data for training. Remove this block to train over entire data
    train_contexts=train_contexts[0:200]
    train_questions=train_questions[0:200]
    train_answers=train_answers[0:200]
    val_contexts=val_contexts[0:200]
    vl_questions=val_questions[0:200]
    val_answers=val_answers[0:200]
    
    #tokenize our context/question pairs.
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)    
    
    # convert our character start/end positions to token start/end positions
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)    
    
    train_dataset = SquadDataset(train_encodings)
    test_dataset = SquadDataset(val_encodings)

    # download model from model hub
    model = DistilBertForQuestionAnswering.from_pretrained(model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        #compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # Saves the model to s3
    trainer.save_model(args.model_dir)