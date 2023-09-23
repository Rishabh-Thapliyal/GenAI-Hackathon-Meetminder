############################################## CONFIG ##############################################

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
file_path = "agenda_list.txt"

#############################################    IMPORTS  ######################################################

# from config import text_gen_model

def create_agenda(email_chain_str):
    '''
    This function reads a email chain and prepares an agenda list.
    Writes the output to a txt file
    '''
    agenda_creator_prompt = "Generate a focused agenda from email chains, extracting key discussion topics, proposed items, \
                        and attachments for a well-structured outline. Maintain clarity and conciseness while \
                        highlighting the meeting's purpose.\
                        Email Chain: \n {email_chain}\
                        Agenda as numbered points:\
                        "
    # Get the model's agenda summarization
    response = text_gen_model.predict(agenda_creator_prompt.format(email_chain = email_chain_str), 
        **parameters
    )
    
    # Write the agenda list to a txt file (to check)
    with open(file_path, "w") as file:
        file.write(response.text)

    # Return the response from the model (to check)
    return response.text


if __name__ == "__main__":
    with open("sample_email_chain.txt", "r") as input_file:
        email_chain_str = input_file.read()

    create_agenda(email_chain_str)