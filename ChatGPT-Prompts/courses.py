from pyChatGPT import ChatGPT
import os

class ChatBot(object):

    def __init__(self, session_token):
        self.session_token = session_token
        self.init_bot()

    def init_bot(self):
        self.bot = ChatGPT(self.session_token)
    
    def get_request(self, message):
        response = self.bot.send_message(message)
        return response['message']


#  courses_list = ['Computer Graphics', 'Quantum Computing', 'Blockchain', 'Cryptography']
#  courses_list = ['Computer Vision', 'Deep Learning', 'Deep Reinforcement Learning', 'Natural Language Processing']
#  courses_list = ['Robotics', 'Machine Learning', 'Bioinformatics', 'Data Mining', 'Data Science']
courses_list = ['Number Theory', 'Differential Equations', 'Signal Processing', 
        'Complex Analysis']
courses_list = ['Probability Theory', 'Descriptive Statistics',
    'Inferential Statistics', 'Linear Regression', 'Time Series Analysis', 
    'Bayesian Statistics', 'Data Visualization']
courses_list = ['Investments', 'Stochastic Processes', 'Corporate Finance', 
    'Microeconomics', 'Macroeconomics', 'Life Contingencies', 'Casualty Contingencies']
courses_list = ['Financial Market', 'Fixed Income Securities', 'Portfolio Theory', 
    'Equities and Options', 'Risk Management', 'Econometrics', 'Fiancial Products', 
    'Credit Risk Models', 'Foreign Exchange Models', 'Commodity Markets']


def main():
    # init session token
    token_path = "ChatGPT-Prompts/session_token.txt"
    with open(token_path, 'r') as f:
        session_token = f.read().strip()

    # prepare output directory to save prompt in
    output_dir = "ChatGPT-Prompts/outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create bot
    bot = ChatBot(session_token)

    # make request from ChatGPT
    for course in courses_list:
        # get request
        message = f'generate complete curriculum for {course}'
        response = bot.get_request(message)

        # save response to markdown file
        output_path = os.path.join(output_dir, f'{course} Curriculum.md')
        with open(output_path, 'w') as f:
            title = f"# {course} Curriculum\n"
            f.write(title)
            f.write('\n')
            f.write(response)
    

if __name__ == "__main__":
    main()
