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


#  degrees_list = ['Pure Maths', 'Statistics']
#  degrees_list = ['Actuarial Science', 'Quantitative Finance', 'Bioinformatics']
degrees_list = ['Physics', 'Chemistry', 'Economy', 'History', 'Psychology', 
                'Philosophy', 'Law', 'Medicine', 'Pharmacy']

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
    for degree in degrees_list:
        # get request
        message = f'generate complete curriculum for {degree}. Include all courses chapter, topics and description'
        response = bot.get_request(message)

        # save response to markdown file
        output_path = os.path.join(output_dir, f'ChatGPT - {degree} Curriculum.md')
        with open(output_path, 'w') as f:
            title = f"# {degree} Curriculum\n"
            f.write(title)
            f.write('\n')
            f.write(response)
    

if __name__ == "__main__":
    main()
