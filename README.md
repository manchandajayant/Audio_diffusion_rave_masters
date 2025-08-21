Download stable audio 1.0

# Use git clone to get the repository structure and LFS pointers

# This will take quite a while

!git clone https://huggingface.co/stabilityai/stable-audio-open-1.0

The above might prompt for your huggingface key, get it from huggingface

Install packages :
pip install -r requirements.txt

## After you have downloaded stable audio, you will have to make a small change in its config

stable-audio-open-1.0 -> tokenizer -> tokenizer_config.json

Edit model_max_length in to 64

Save and run

Run the notebook

Change hte input and style path names as you like
# Audio_diffusion_rave_masters
