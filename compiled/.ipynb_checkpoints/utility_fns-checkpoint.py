import vertexai
from vertexai.language_models import TextGenerationModel
import pandas as pd
import numpy as np
import webvtt
from tqdm import tqdm
import time
import re
import warnings
warnings.filterwarnings('ignore')


p = {}
vertexai.init(project="wmt-7fbls2a91f025anb93e025b02g", location="us-central1")
parameters = {
    "max_output_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
text_gen_model = TextGenerationModel.from_pretrained("text-bison@001")


def create_bins(max_cumm_secs, window_size):
    if int(np.ceil(max_cumm_secs/window_size))!=max_cumm_secs//window_size:
        bins = np.arange(0, max_cumm_secs+window_size, window_size)
    else:
        bins = np.arange(0, max_cumm_secs, window_size)
    labels = 5 * np.arange(1, len(bins))
    return bins, labels


def output_agenda_check(x, agenda_list_cleaned):
    a = set(x['agendas_covered'].tolist())
    b = set(agenda_list_cleaned)
    c = list(a.intersection(b))
    for i in agenda_list_cleaned:
        if i in c:
            p[i] = True
    print('--------------------------------')
    print(f'Agenda Covered in last {x.binned.iloc[0]} mins ==>')
    print(list(p.keys()))
    time.sleep(1)
    print('--------------------------------')
    print('\n')
    

def task1(email_chain_path):
    """
    This function reads a email chain and prepares an agenda list.
    Writes the output to a txt file
    """
    out_file_path = "agenda_list.txt"
    
    with open(email_chain_path, "r") as input_file:
        email_chain_str = input_file.read()

    agenda_creator_prompt = "Generate a focused agenda from email chains, extracting key discussion topics, proposed items, \
                        and attachments for a well-structured outline. Maintain clarity and conciseness while \
                        highlighting the meeting's purpose.\
                        Email Chain: \n {email_chain}\
                        Agenda as numbered points:\
                        "
    # Get the model's agenda summarization
    response = text_gen_model.predict(agenda_creator_prompt.format(email_chain=email_chain_str), **parameters)

    # Write the agenda list to a txt file (to check)
    with open(out_file_path, "w") as file:
        file.write(response.text)
    print('-------------')
    print('Agenda List')
    print('-------------')
    print(response.text)
    print('\n')
    # Return the response from the model (to check)
    return response.text


def task2(zoom, agenda_path):

    transcript_dict = {'start_time': [],
                       'end_time': [],
                       'text': []
                       }
    for caption in webvtt.read(zoom):
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
    transcript_df['cumm_secs'] = transcript_df['seconds'] + transcript_df['minutes'] * 60 + transcript_df[
        'hours'] * 60 * 60

    # Read the agenda list
    with open(agenda_path, "r") as ip_file:
        agenda_list = ip_file.read().split("\n")

    agenda_list_cleaned = [re.sub('[0-9]', '', i) for i in agenda_list]
    agenda_list_cleaned = [word.replace(".", "").strip() for word in agenda_list_cleaned]

    generation_model = TextGenerationModel.from_pretrained("text-bison@001")

    # Prompt
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
    
    print('Processing Zoom Transcript...')
    print('\n')
    # try:
    #     result_df_genAI = pd.read_csv("transcript_agenda_check_genAI.csv")
    # except:
    for secs in tqdm(range(0, transcript_df['cumm_secs'].max(), window_size)):

        conversation_to_be_processed = transcript_df[(transcript_df['cumm_secs'] >= secs) &
                                                     (transcript_df['cumm_secs'] <= secs + window_size)]['text'].str.cat(sep="\n")

        if secs == 0:  # it is the starting lines
            prompt = agenda_checker_prompt.format(context="", agenda_list=agenda_list_cleaned,
                                                  input_to_be_processed=conversation_to_be_processed)

        else:  # for summarising later convos, take previous summaries as context
            prompt = agenda_checker_prompt.format(
                context=conversation_covered[- 1], agenda_list=agenda_list_cleaned,
                input_to_be_processed=conversation_to_be_processed
            )

        agenda = generation_model.predict(prompt=prompt, max_output_tokens=1024).text

        agendas_covered.append(agenda)
        conversation_covered.append(conversation_to_be_processed)
        time_stamp.append(secs + window_size)

    result_df_genAI = pd.DataFrame(data={'time_passed': time_stamp,
                                         'summary': conversation_covered,
                                         'agendas_covered': agendas_covered,
                                         })

    max_cumm_secs = result_df_genAI['time_passed'].describe().loc['max']
    window_size = 300
    bins, labels = create_bins(max_cumm_secs, window_size)

    result_df_genAI['binned'] = pd.cut(result_df_genAI['time_passed'], bins=bins, labels=labels)
    result_df_genAI['binned'] = result_df_genAI['binned'].astype(int)
    result_df_genAI.to_csv("transcript_agenda_check_genAI.csv")

    result_df_genAI.groupby('binned').apply(lambda x: output_agenda_check(x, agenda_list_cleaned))
    



def task3(zoom):

    transcript_dict = {'start_time': [],
                       'end_time':[],
                       'text': []
                      }
    for caption in webvtt.read(zoom):
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
    print('\n')
    print('Creating Minutes of Meeting....')
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
    print('\n')
    mom_creator_prompt_ = mom_creator_prompt.format(context="", input_to_be_processed=transcript_summary)
    print(generation_model.predict(prompt=mom_creator_prompt_, max_output_tokens=1024).text)


