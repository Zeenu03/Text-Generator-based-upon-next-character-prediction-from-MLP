import streamlit as st

import numpy as np
import time
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.write("#NN based Text Generator")

st.sidebar.title("Model Information")

st.sidebar.write("This is a simple Neural Network based text generator. It can predict next characters of the input text provided. The model uses lower case letters, punctuation marks and fullstops. The text is generated paragraph wise, because the model learnt this from the text corpus.")

num_chars = st.slider("Number of characters to be generated", 100, 2000, 1000)
input_blocksize= st.selectbox(
    'Select Model with',
    ('Block_size=10  Emb Size=60', 'Block_size=10  Emb Size=150','Block_size=100  Emb Size=60','Block_size=100  Emb Size=150')
)

with open("startup_funding.txt", 'r') as file:
    thefile = file.read()

sort_char = ''

for char in thefile:
    if char in ['\x0c','\ufeff','»', '¿', 'â', 'ï', '”','€']:
        continue
    sort_char += char.lower()

characters = sorted(list(set(sort_char)))

stoi = {s: i+1 for i, s in enumerate(characters)}
stoi['~'] = 0 # for padding
stoi['—'] = 53
itos = {i : s for s,i in stoi.items()}
# print(stoi)
def generate_text(model,itos,stoi, block_size, max_len, start_str = None):
    g = torch.Generator()
    g.manual_seed(42)
    context = [0]*block_size
    if start_str:
        for char in start_str:
            context = context[1:] + [stoi[char]]
    text = start_str if start_str else ''
    for _ in range(max_len):
        x = torch.tensor(context).view(1,-1).to(device)
        y_pred = model(x)
        ix = torch.distributions.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        text += ch
        context = context[1:] + [ix]
    return text

def type_text(text):
    text_element = st.empty()
    s = ''
    for i in text:
        s += i
        text_element.write(s + '$I$')
        time.sleep(0.005)
    text_element.write(s)

class NextChar(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.activation = nn.ReLU()  # Adding ReLU activation function
        self.dropout = nn.Dropout(0.2)  # Adding dropout with 20% probability
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))  # Applying ReLU activation
        x = self.dropout(x)  # Applying dropout
        x = self.lin2(x)
        return x

# print(len(stoi))

# model = torch.compile(model)

if(input_blocksize == 'Block_size=10  Emb Size=60'):
    block_size = 10
    emb_dim=60
elif(input_blocksize == 'Block_size=10  Emb Size=150'):
    block_size = 10
    emb_dim=150
elif(input_blocksize == 'Block_size=100  Emb Size=60'):
    block_size = 100
    emb_dim=60
else:
    block_size = 100
    emb_dim=150
    
input_text = st.text_input("Enter Text", placeholder="Enter a text (In Lower Case) or leave it empty")

model = NextChar(block_size, len(stoi), emb_dim, 512).to(device)
# if input text contains any character other than the ones in the model, then throw an error
for char in input_text:
    if char not in characters:
        st.error("Invalid Characters in the input text")
        break

btn = st.button("Generate Text")

if btn:
    st.subheader("Seed Text")
    type_text(input_text)
    if(input_blocksize == 'Block_size=10  Emb Size=60'):
        model.load_state_dict(torch.load("model_story11.pth", map_location = device))
    elif(input_blocksize == 'Block_size=10  Emb Size=150'):
        model.load_state_dict(torch.load("model_story12.pth", map_location = device))
    elif(input_blocksize == 'Block_size=100  Emb Size=60'):
        model.load_state_dict(torch.load("model_story21.pth", map_location = device))
    else:
        model.load_state_dict(torch.load("model_story22.pth", map_location =device))
        
    gen_text = generate_text(model, itos, stoi, block_size, num_chars, input_text)
    st.subheader("Generated Text")
    print(gen_text)
    type_text(gen_text)