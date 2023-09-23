from utility_fns import task1, task2, task3


class Agent:
    """
    This function inputs one of the following or more
    Task #1 (T1): Enter email which needs to be summarized and the agenda created
    Task #2 (T2): Enter zoom transcripts to check if agendas are covered or not
    Task #3 (T3): Enable mom to True and provide email and zoom to create MoMs

    Parameters
    ----------
    email : str
        Email chains are used to create agendas using T1. It uses GenAI
        based summarization.

    zoom : str, default=''
        Zoom transcripts are utilized to check if the agendas are completed
        for the specific Zoom call or not in T2. Also in T3 it is used
        to create MoMs.

    mom : bool, default=False
        Minutes of meeting can be created by enable this option to True.
        Zoom transcript needs to be provided here

    """
    def __init__(self, email='', zoom='', mom=False, task='ALL'):
        self.email = email
        self.zoom = zoom
        self.mom = mom
        self.task = task

    @staticmethod
    def run_task_one(email):
        agenda_lst = task1(email)
        return agenda_lst

    @staticmethod
    def run_task_two(zoom, agenda_lst):
        task2(zoom, agenda_lst)

    @staticmethod
    def run_task_three(zoom):
        task3(zoom)

    def run_app(self):
        if self.task.upper() == 'T1':
            agenda_lst = self.run_task_one(self.email)
            self.agenda_lst = agenda_lst
        elif self.task.upper() == 'T2':
            self.run_task_two(zoom=self.zoom, agenda_lst=self.agenda_lst)
        elif self.task.upper() == 'T3':
            self.run_task_three(zoom=self.zoom)
        elif self.task.upper() == 'ALL':
            agenda_lst = self.run_task_one(self.email)
            self.agenda_lst = agenda_lst
            df = self.run_task_two(zoom=self.zoom, agenda_lst=self.agenda_lst)
            self.run_task_three(zoom=self.zoom)


obj = Agent(email='sample_email_chain.txt', zoom='meeting002_transcript.vtt')
obj.run_app()
