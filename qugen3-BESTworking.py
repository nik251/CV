import openai
import nltk
from nltk.tokenize import sent_tokenize

# Set up the OpenAI API client
openai.api_key = "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"


with open("text.txt", 'r', encoding="utf8") as file:
        text_file = file.read()


text = str(text_file)
number_of_sentences = sent_tokenize(text)
len_text = (int(len(number_of_sentences)))
num = len_text//10
#print(num)

    ## each prompt can only be 10 sentences long
x = 0
y = 10





# Set up the model and prom
model_engine = "text-davinci-003"


def gen_mcq():
       prompt = ("please create MCQ based on this text:" + extract)
       completion = openai.Completion.create(
               engine=model_engine,
                prompt=prompt,  
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
                )
       response = completion.choices[0].text
       print(response)
       prompt = ("what is the correct answer for:" + response)
       completion = openai.Completion.create(
               engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
                )
       response = completion.choices[0].text
       correct_answer = response
       print(correct_answer)

def gen_lnqu():
        prompt = ("please create a long answer question based on this text:" + extract)
        completion = openai.Completion.create(
                engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
                )
        response = completion.choices[0].text
        print(response)
        prompt = ("what is the correct answer for: " + response + " based on " + extract)
        completion = openai.Completion.create(
               engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
                )
        response = completion.choices[0].text
        correct_answer = response
        print(correct_answer)



def gen_flash():
       prompt = ("please create a short title for this paragraph" + extract)
       completion = openai.Completion.create(
               engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
                )
       response = completion.choices[0].text
       print(response)
       prompt = ("please paraphrase this information:" + extract)
       completion = openai.Completion.create(
               engine=model_engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
                )
       response = completion.choices[0].text
       print(response)

for i in range(num+1):
        extract = str(number_of_sentences[x:y])
        #print(extract)
        x = x + 10
        y = y + 10
        gen_flash()
        gen_lnqu()
        gen_mcq()