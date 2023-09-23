import vertexai
from vertexai.language_models import TextGenerationModel

vertexai.init(project="wmt-7fbls2a91f025anb93e025b02g", location="us-central1")
parameters = {
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
text_gen_model = TextGenerationModel.from_pretrained("text-bison@001")
