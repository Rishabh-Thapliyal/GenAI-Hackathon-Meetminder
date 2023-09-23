import pandas as pd
import webvtt
from tqdm import tqdm
import re
import warnings
import vertexai
from vertexai.language_models import TextGenerationModel
warnings.filterwarnings('ignore')


transcript_dict = {'start_time': [],
                   'end_time':[],
                   'text': []
                  }
for caption in webvtt.read('meeting002_transcript.vtt'):
    transcript_dict['start_time'].append(caption.start)
    transcript_dict['end_time'].append(caption.end)
    transcript_dict['text'].append(caption.text)

transcript_df = pd.DataFrame(transcript_dict)
transcript_df['text'] = transcript_df['text'].apply(lambda x: x.strip())
transcript_df['start_time'] = pd.to_datetime(transcript_df['start_time'])
transcript_df['end_time'] = pd.to_datetime(transcript_df['end_time'])
transcript_df['hours'] = transcript_df['start_time'].dt.strftime('%H').astype('int')
transcript_df['minutes'] = transcript_df['start_time'].dt.strftime('%M').astype('int')
transcript_df['seconds'] = transcript_df['start_time'].dt.strftime('%S').astype('int')
transcript_df['cumm_secs'] = transcript_df['seconds']+ transcript_df['minutes']*60 + transcript_df['hours']*60*60


vertexai.init(project="wmt-7fbls2a91f025anb93e025b02g", location="us-central1")
parameters = dict(max_output_tokens=1024, temperature=0.2, top_p=0.8, top_k=40)

# Read the agenda list
with open("../final_codes/agenda_list.txt", "r") as ip_file:
    agenda_list = ip_file.read().split("\n")


agenda_list_cleaned = [re.sub('[0-9]', '', i) for i in agenda_list]
agenda_list_cleaned = [word.replace(".", "") for word in agenda_list_cleaned]

generation_model = TextGenerationModel.from_pretrained("text-bison@001")

## Prompt
agenda_checker_prompt = """
        Taking the following context delimited by triple backquotes into consideration:
        ```{context}```
        Agenda points to be discussed during the call are listed below delimited by triple backquotes:
        ```{agenda_list}```
        Conversation transcript is given below delimited by triple backquotes:  
        ```{input_to_be_processed}```
          You are given a agenda list and a piece of conversation transcript. 
          Understand clearly the agenda points and identify which of the agenda points from the list is being discussed in the above conversation.
          
          If its a topic not mentioned in the agenda, say "General conversation".
          Return only the agenda from the above agenda list provided.
    """


agendas_covered = []
conversation_covered = []
time_stamp = []
window_size = 30

for secs in tqdm(range(0, transcript_df['cumm_secs'].max(), window_size)):

    conversation_to_be_processed = transcript_df[(transcript_df['cumm_secs']>= secs) & (transcript_df['cumm_secs']<= secs+window_size)]['text'].str.cat(sep ="\n")

    if secs == 0:  # it is the starting lines
        prompt = agenda_checker_prompt.format(context="",agenda_list=agenda_list_cleaned,
                                              input_to_be_processed=conversation_to_be_processed)

    else:  # for summarising later convos, take previous summaries as context
        prompt = agenda_checker_prompt.format(
            context=conversation_covered[- 1],agenda_list=agenda_list_cleaned,
            input_to_be_processed=conversation_to_be_processed
        )

    agenda = generation_model.predict(prompt=prompt, max_output_tokens=1024).text

    agendas_covered.append(agenda)
    conversation_covered.append(conversation_to_be_processed)
    time_stamp.append(secs+window_size)

result_df_genAI= pd.DataFrame(data = {'time_passed':time_stamp,
                'summary':conversation_covered,
                'agendas_covered':agendas_covered,
                        })

result_df_genAI.to_csv("results_df_genAI.csv")
