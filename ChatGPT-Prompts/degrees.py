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
#  degrees_list = ['Chemistry', 'Economy', 'History', 'Psychology', 
#                  'Philosophy', 'Law', 'Medicine', 'Pharmacy']
#  degrees_list = ['Neurosciences', 'Physics']
degrees_list = ['Cybersecurity']

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
        #  message = f"generate complete curriculum for {degree}. Include all courses chapter, topics, description, explanation and formula. For each chapter, please provide a short explanation of the most important concept and a formula to accompany it. If you can, list all the possible subfields. Be as precise as possible and provide more information. Find corresponding courses website from CMU, MIT, Stanford, Berkley and other Ivy league universities. Suggest ressources to use to learn and solutions.  Example: - Course: Real Analysis - Curriculum: - Video Lecture: - Website: https://www.stat.cmu.edu/~hseltman/309/Book/"
        message = f"generate complete curriculum for {degree} and subfields"
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
