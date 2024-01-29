import openai
from time import sleep
openai.api_key = open("key.txt", "r").read().strip("\n")

import json
f = open('train.json')
input = json.load(f)

output = []
import os.path
if os.path.isfile('data.json'):
    f2 = open('data.json')
    output = json.load(f2)
else:
    output = []

i = len(output)
limit = 0
while i < len(input):
  sp = r"System Prompt: given claim text and evidence text, determine the probabilities that the evidence refutes, supports and is neutral against the claim. Finish with either the {Support} {Refute} or {Neutral} label on the last line."
  message = sp + " Claim: " + input[i]['claim'] + " Evidence: " + input[i]['evidence']
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = [{"role": "user", "content": message[:16000]}]
  )

  respond = completion.choices[0].message.content[-100:]
  print(respond)

  if limit % 3 == 0:
     limit = 0
     print("Waiting 1 minute for API rate limit...")
     sleep(60)

  front = respond.find('{')
  end = respond.find('}')

  if (front != -1 and end != -1):
    result = respond[front + 1:end]
    output.append(result)
    with open('data.json', 'w') as f:
      json.dump(output, f)
    print(i, ": ", result)
  else:
     print(i, ": Not Found")
     i -= 1
     
  i += 1
  limit += 1