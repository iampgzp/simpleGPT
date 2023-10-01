<div align="center">

# Simple GPT

### Simple implementation of the Transformer Decoder
</div>

1. Implemented in One File. Less than 300 lines of code.
2. It's the architecture used in GPT and described in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. The code is extensively documented with annotations to help people understand

## How To Use

</div>

1. Put the text files you want to train on into the data folder. (Remove the Tiny-Shakespear.txt sample data file)
2. Ran "pip install -r requirements.txt" to install dependencies. (Preferrably in a Python Virtual Environment)
3. Ran "Python3 gpt.py"

That's it. If it's successful, it will print the training and evaluation loss until the end which it will print a sample text generation. For example:

![Screenshot 2023-09-30 at 5 04 07 PM](https://github.com/iampgzp/simpleGPT/assets/5464549/dccc0eb6-4617-4542-a650-7302ab84ee84)


<div align="center">

## Model Architecture

</div>

<img width="445" alt="image" src="https://github.com/iampgzp/simpleGPT/assets/5464549/1fec1d37-fb1a-4ed7-8d8b-e2f97fef3d87">


<div align="center">

## Further Reading

</div>

1. The project is based on this awesome youtube [tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&themeRefresh=1) from Andrej Karpathy. I fixed some minor bugs so that it can ran smoothly
2. The model with the current hyperparameters has roughly 10M parameters (It's related to the size of data vocab size as well) with dtype = long, it will take roughly 80MB of memory
3. Running with an A100 GPU on Google Colab, on the Tiny Shakespear data set, it roughly take 10 mins to ran 5000 training iterations
4. To get better result, a better choice will be to start from a pre-trained foundational model. But the purpose of this repo is to help people build and understand the core of Transformer
5. Lastly, this is only the decoder section of transformer. If you are interested in a ChatGPT Style, sequence to sequence model, we will need to implement the encoder. 
