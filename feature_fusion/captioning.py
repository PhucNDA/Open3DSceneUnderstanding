import torch
from promptcap import PromptCap
from promptcap import PromptCap_VQA

model = PromptCap("vqascore/promptcap-coco-vqa")  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"
# vqa_model = PromptCap_VQA(promptcap_model="vqascore/promptcap-coco-vqa", qa_model="allenai/unifiedqa-t5-base")
if torch.cuda.is_available():
  model.cuda()
#   vqa_model.cuda()

def query(image, target):
    question = "Is it a "+target+" ?"
    print(model.caption(question, image))
    question = "Where is the "+ target+" ?"
    print(model.caption(question, image))

breakpoint()
target = 'Sofa'
image = "zztest/42446100_77965.457.png"
query(image, target)