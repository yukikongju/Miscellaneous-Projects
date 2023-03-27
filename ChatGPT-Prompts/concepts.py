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

#  concept_list = ['Real Analysis']
#  concept_list = ['Combinatorics', 'Graph Theory', 'Computation Theory', 
#      'Compilers', 'Operating Systems', 'Distributed Systems', 'Computer Architecture', 
#      'Interpreters', 'Linear Regression']


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
    for concept in concept_list:
        # get request
        message = f"List all importants concepts (definitions, theorems) in {concept}. Explain big picture of the field"
        response = bot.get_request(message)

        # save response to markdown file
        output_path = os.path.join(output_dir, f'ChatGPT Concept Explained - {concept}.md')
        with open(output_path, 'w') as f:
            title = f"# {concept} Curriculum\n"
            f.write(title)
            f.write('\n')
            f.write(response)


if __name__ == "__main__":
    main()
