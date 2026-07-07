import os
import re
import torch

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)


# ==========================================================
# INFLUENCE SPACE MARKOV
# ==========================================================

class InfluenceSpaceMarkov:

    def __init__(
        self,
        beta=2.0,
        alpha=3.0,
        gamma=2.0
    ):
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma


    def fit(self, text):

        words = re.findall(
            r"\b\w+\b",
            text.lower()
        )

        self.vocab = sorted(
            set(words)
        )

        self.word_to_idx = {
            w:i
            for i,w in enumerate(self.vocab)
        }

        size = len(self.vocab)

        counts = torch.zeros(
            (size,size),
            dtype=torch.float32
        )


        for a,b in zip(
            words[:-1],
            words[1:]
        ):

            i = self.word_to_idx[a]
            j = self.word_to_idx[b]

            counts[i,j] += 1



        L = torch.log1p(counts)


        Y = torch.exp(
            self.beta * L
            +
            torch.sin(
                self.gamma * L
            )
        )


        Y[counts == 0] = 0


        maximum = max(
            Y.max().item(),
            1
        )


        Y /= maximum


        Z = torch.exp(
            self.alpha * Y
            +
            torch.sin(
                self.gamma * Y
            )
        )


        Z[Y == 0] = 0


        row_sum = Z.sum(
            dim=1,
            keepdim=True
        )


        self.P = torch.where(
            row_sum != 0,
            Z / row_sum,
            torch.zeros_like(Z)
        )


        return self



    def transform(self,text):

        words = re.findall(
            r"\b\w+\b",
            text.lower()
        )


        result=[]


        for word in words:

            if word not in self.word_to_idx:
                continue


            idx = self.word_to_idx[word]

            weights = self.P[idx]


            if weights.sum() == 0:

                result.append(word)

            else:

                next_word_id = torch.argmax(
                    weights
                ).item()

                result.append(
                    self.vocab[next_word_id]
                )


        return " ".join(result)



# ==========================================================
# PYTORCH DATASET
# ==========================================================

class TextDataset(Dataset):

    def __init__(
        self,
        texts,
        tokenizer
    ):

        self.data = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256
        )


    def __len__(self):

        return len(
            self.data["input_ids"]
        )


    def __getitem__(
        self,
        index
    ):

        item = {

            key:
            torch.tensor(
                value[index]
            )

            for key,value
            in self.data.items()

        }


        item["labels"] = (
            item["input_ids"].clone()
        )


        return item



# ==========================================================
# BUILD INFLUENCE DATASET
# ==========================================================

def build_dataset(
    filename,
    chunk_size=256
):

    print(
        "Loading dataset..."
    )


    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as f:

        text=f.read()



    print(
        "Building influence matrix..."
    )


    influence = InfluenceSpaceMarkov()

    influence.fit(
        text
    )


    words=text.split()


    chunks=[]


    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk=" ".join(
            words[
                i:i+chunk_size
            ]
        )


        changed=influence.transform(
            chunk
        )


        chunks.append(
            changed
        )


    print(
        "Samples:",
        len(chunks)
    )


    return chunks



# ==========================================================
# TRAIN MODEL
# ==========================================================

def train():

    model_name = (
        "HuggingFaceTB/"
        "SmolLM2-135M-Instruct"
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )


    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )



    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )



    texts = build_dataset(
        "singlekb.txt"
    )


    dataset = TextDataset(
        texts,
        tokenizer
    )



    print(
        "Dataset size:",
        len(dataset)
    )



    args = TrainingArguments(

    output_dir="./influenced_model",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-5,

    logging_steps=1,

    save_strategy="epoch",

    save_total_limit=2,

    report_to=[]

)



    trainer = Trainer(

        model=model,

        args=args,

        train_dataset=dataset

    )


    print(
        "TRAINING..."
    )


    trainer.train()



    print(
        "SAVING MODEL..."
    )


    trainer.save_model(
        "./influenced_model"
    )


    tokenizer.save_pretrained(
        "./influenced_model"
    )


    print(
        "DONE"
    )



# ==========================================================
# CHAT
# ==========================================================

def chat():

    path="./influenced_model"


    if not os.path.exists(path):

        print(
            "Train the model first."
        )

        return



    tokenizer = AutoTokenizer.from_pretrained(
        path
    )


    model = AutoModelForCausalLM.from_pretrained(
        path
    )


    model.eval()



    while True:


        prompt=input(
            "USER: "
        )


        if prompt.lower() in [
            "exit",
            "quit"
        ]:

            break



        inputs=tokenizer(
            prompt,
            return_tensors="pt"
        )



        with torch.no_grad():

            output=model.generate(

                **inputs,

                max_new_tokens=650,

                temperature=0.8,

                top_p=0.95,

                do_sample=True,

                pad_token_id=
                tokenizer.eos_token_id

            )



        print()

        print(
            tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )
        )

        print("-"*80)



# ==========================================================
# MAIN
# ==========================================================

if __name__=="__main__":


    print(
        "1 = Train"
    )

    print(
        "2 = Chat"
    )


    choice=input(
        "> "
    )


    if choice=="1":

        train()


    elif choice=="2":

        chat()
