from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch

# # Set a configuration for our RoBERTa model
# config = RobertaConfig(
#     vocab_size=8192,
#     max_position_embeddings=514,
#     num_attention_heads=12,
#     num_hidden_layers=6,
#     type_vocab_size=1,
# )
# # Initialize the model from a configuration without pretrained weights
# model = RobertaForMaskedLM(config=config)
# print('Num parameters: ',model.num_parameters())

model = RobertaForMaskedLM.from_pretrained('roberta-base')


# Create the tokenizer from a trained one
tokenizer = RobertaTokenizerFast.from_pretrained("code-toknzr-roberta", max_len=512)
print("Tokenizer Initiated!")

class CodeDataset():
    def __init__(self, datafile, tokenizer):
        # or use the RobertaTokenizer from `transformers` directly.
        self.examples = []
        # For every value in the dataframe 
        with open(datafile,'r') as fp:
            for example in fp:
                # 
                x=tokenizer.encode_plus(example, max_length = 512, truncation=True, padding=True)
                self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])
      
# Create the train and evaluation dataset
train_dataset = CodeDataset("train_sent.txt", tokenizer)
eval_dataset = CodeDataset("test_sent.txt", tokenizer)
print("Dataset Initiated!")

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Use Multiple GPUs for Training
use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()
device = torch.device("cuda" if use_cuda else "cpu")
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3], dim=0)
if use_cuda:
    model = model.cuda()
    

# Define the training arguments
training_args = TrainingArguments(
    output_dir='dec4_gpt',
    overwrite_output_dir=True,
    evaluation_strategy = 'epoch',
    num_train_epochs=3,
    learning_rate=1e-4,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    save_steps=4096,
    logging_steps=4096,
    #eval_steps=4096,
    save_total_limit=1,
)

# Create the trainer for our model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #prediction_loss_only=True,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model('./saved_dec7_roberta_a')

# Evaluating on Test data
trainer.evaluate(eval_dataset)