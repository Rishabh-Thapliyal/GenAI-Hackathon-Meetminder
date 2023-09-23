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
for caption in webvtt.read('../rishabh/meeting002_transcript.vtt'):
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


generation_model = TextGenerationModel.from_pretrained("text-bison@001")

summary_prompt_template = """
    Taking the following context delimited by triple backquotes into consideration:

    ```{context}```

    Write a concise summary of the following conversation delimited by triple backquotes. Identify the main points of discussion and 
    any conclusions that are reached. Do not incorporate other general knowledge. Summary is in plain text, in complete sentences, 
    with no markup or tags.

    ```{conversation}```

    CONCISE SUMMARY:
"""

initial_summary = []

window_size = 30

for secs in tqdm(range(0, transcript_df['cumm_secs'].max(), window_size)):
   
    conversation_to_be_summarised = transcript_df[(transcript_df['cumm_secs']>= secs) & (transcript_df['cumm_secs']<= secs+window_size)]['text'].str.cat(sep ="\n")
    
    if secs == 0:  # its the starting lines
        prompt = summary_prompt_template.format(context="", conversation=conversation_to_be_summarised)

    else:  # for summarising later convos, take previous summaries as context
        prompt = summary_prompt_template.format(
            context=initial_summary[- 1], conversation=conversation_to_be_summarised
        )

    # Generate a summary using the model and the prompt
    summary = generation_model.predict(prompt=prompt, max_output_tokens=1024).text

    # Append the summary to the list of summaries
    initial_summary.append(summary)


transcript_summary = " ".join(set(initial_summary))


## Prompt
mom_creator_prompt = """
        Taking the following context delimited by triple backquotes into consideration:
        ```{context}```
        Conversation transcript's summary is given below delimited by triple backquotes:  
        ```{input_to_be_processed}```
        Generate meeting minutes from the above summary of zoom transcript, highlighting key discussions, decisions, and action items for the participants involved in the discussion.
        If you think there are no action items for the participants, action item would be that the host/key speaker will send out the zoom recording and related materials across to the participants.
        Meeting minutes (should be well structures, should contain meeting details, the attendees list, key discussion points, action items in this order):
    """

mom_creator_prompt_ = mom_creator_prompt.format(context="", input_to_be_processed=transcript_summary)

print(generation_model.predict(prompt=mom_creator_prompt_, max_output_tokens=1024).text)

